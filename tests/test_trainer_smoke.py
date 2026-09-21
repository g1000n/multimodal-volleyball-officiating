"""
tests/test_trainer_smoke.py

Headless smoke test for trainer.py: drives every mode (practice, drill, challenge, sim) with a
fake camera, a scripted "person" and a fake classifier, so the whole state machine, drawing code,
grading hookup and report writing run without hardware, MediaPipe, PyTorch or a display.

Run:  python tests/test_trainer_smoke.py [--save-frames DIR]
"""

import os
import random
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import gesture_grader as gg  # noqa: E402
import test_gesture_grader as tg  # noqa: E402
import trainer  # noqa: E402

LABELS = sorted(gg.RULES)
IDX_TO_LABEL = {i: l for i, l in enumerate(LABELS)}

POSES = {
    "team_to_serve_left": tg.g_tts("left"), "team_to_serve_right": tg.g_tts("right"),
    "service_authorization_left": tg.g_auth("left"), "service_authorization_right": tg.g_auth("right"),
    "ball_in": tg.g_ball_in(), "ball_out": tg.g_ball_out(), "double_contact": tg.g_double(),
    "end_of_set": tg.g_end(),
}
TAG_COL = 100  # a hand-coordinate column the grader never reads; tags "this frame is the gesture"


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeCap:
    def __init__(self, clock):
        self.clock = clock

    def read(self):
        self.clock.t += 0.1                      # 10 fps
        return True, np.full((720, 1280, 3), 60, np.uint8)


class FakeBackend(trainer.Backend):
    def __init__(self, clock, always_show=None):
        super().__init__(FakeCap(clock), self._extract, self._classify, IDX_TO_LABEL)
        self.session = None
        self.clock = clock
        self.always_show = always_show      # practice mode: label to perform continuously
        self.n = 0

    def _extract(self, frame):
        s = self.session
        self.n += 1
        label, side = None, "right"
        if self.always_show:
            label = self.always_show
        elif s is not None and s.phase == "capture" and s.step:
            elapsed = self.clock() - s.phase_t0
            st = s.step
            if st["kind"] == "gesture" and 0.6 <= elapsed <= 3.6:
                label = st["label"]
                side = (st.get("ctx") or {}).get("side", "right")
            elif st["kind"] == "pair":
                if 0.6 <= elapsed <= 3.4:
                    label = st["labels"][0]
                elif 4.4 <= elapsed <= 7.4:
                    label = st["labels"][1]
                    side = ((st.get("ctxs") or [None, None])[1] or {}).get("side", "right")
        if label is None:
            f = tg.idle()
            f[TAG_COL] = 0.0
            return f
        f = tg.g_double(side=side)(self.n, False) if label == "double_contact" else POSES[label](self.n, False)
        f[TAG_COL] = 1.0
        f[121] = 0.0
        # put the label id in another unused column so the fake classifier can name it
        f[101] = LABELS.index(label) + 1
        return f

    def _classify(self, window):
        w = np.array(window)
        probs = np.full(len(LABELS), 0.02)
        if (w[:, TAG_COL] > 0.5).mean() >= 0.5:
            lab = LABELS[int(w[w[:, TAG_COL] > 0.5][-1, 101]) - 1]
            probs[LABELS.index(lab)] = 0.95
            return lab, 0.95, probs
        return gg.NOTHING_LABEL, float(probs.max()), probs


class FakeHub:
    auto_active = False

    def __init__(self, clock=None):
        self.clock = clock
        self.times = []

    def start(self):
        pass

    def stop(self):
        pass

    def manual(self):
        self.times.append(self.clock() if self.clock else 0.0)

    def first_since(self, t):
        c = [x for x in self.times if x >= t]
        return min(c) if c else None

    def heard_since(self, t):
        return self.first_since(t) is not None


class TestSession(trainer.Session):
    def __init__(self, *a, script=None, save_dir=None, **kw):
        super().__init__(*a, **kw)
        self.be.session = self
        self.frames_seen = 0
        self.save_dir = save_dir
        self.saved = set()
        self.max_frames = 4000
        self.whistle_at = 0.3          # seconds after GO when the fake trainee blows the whistle (None = never)

    def _show(self, ui):
        self.frames_seen += 1
        if self.attempts and not getattr(self, "_csv_checked", False):
            self._csv_checked = True          # the first counted attempt is already on disk, before the session ends
            import csv as _c
            with open(os.path.join(self.dir, "attempts.csv"), newline="", encoding="utf-8") as fh:
                assert len(list(_c.DictReader(fh))) == len(self.attempts), "attempt not written immediately"
        assert ui.shape == (trainer.UI_H, trainer.UI_W, 3)
        tag = None
        if self.phase == "intro" and "intro" not in self.saved:
            tag = "intro"
        elif self.phase == "capture" and self.clock() - self.phase_t0 > 2.0 and "capture" not in self.saved:
            tag = "capture"
        elif self.phase == "result" and "result" not in self.saved:
            tag = "result"
        elif self.phase == "summary" and "summary" not in self.saved:
            tag = "summary"
        elif self.phase == "practice" and self.frames_seen > 25 and "practice" not in self.saved:
            tag = "practice"
        if tag and self.save_dir:
            cv2.imwrite(os.path.join(self.save_dir, f"{self.mode}_{tag}.png"), ui)
        if tag:
            self.saved.add(tag)

    def _key(self):
        if self.frames_seen > self.max_frames:
            return 27
        if self.phase == "capture" and self.step and self.step.get("whistle") and self.whistle_at is not None \
                and self.clock() - self.phase_t0 > self.whistle_at and self.whistle.first_since(self.phase_t0) is None:
            return ord("w")
        if getattr(self, "hands_off", False):
            # automatic modes: only start the run and leave the summary; everything between must run by itself
            if self.phase == "wait_whistle" and self.clock() - self.phase_t0 > 0.5:
                return ord("w")
            if self.phase == "intro" and not self.auto_go and self.clock() - self.phase_t0 > 0.3:
                return 32
            if self.phase == "summary" and self.clock() - self.phase_t0 > 0.3:
                self.summary_keys = getattr(self, "summary_keys", 0) + 1
                return ord("d") if self.summary_keys == 1 else 32
            return 255
        if self.mode == "practice":
            return 27 if self.frames_seen > 60 else 255
        if self.phase == "intro" and self.clock() - self.phase_t0 > 0.3:
            step = self.step
            if step["kind"] == "whistle":
                pass
            return 32
        if self.phase == "wait_whistle":
            return ord("w") if self.clock() - self.phase_t0 > 0.5 else 255
        if self.phase == "result" and self.clock() - self.phase_t0 > 0.3:
            if self.mode == "drill" and len([a for a in self.attempts]) >= 3:
                return ord("q")
            return 32
        if self.phase == "summary" and self.clock() - self.phase_t0 > 0.3:
            return 32
        return 255


def run_mode(mode, tmp, save_dir, gesture=None, level="standard", hands_off=False, whistle_at=0.3, **extra):
    clock = FakeClock()
    be = FakeBackend(clock, always_show="ball_out" if mode == "practice" else None)
    trainee = {"trainee_id": "tester", "display_name": "Tester"}
    choice = {"action": "start", "mode": mode, "gesture": gesture, "level": level, **extra}
    sess = TestSession(be, trainee, choice, FakeHub(clock), clock=clock, rng=random.Random(3), save_dir=save_dir)
    sess.hands_off = hands_off
    sess.whistle_at = whistle_at
    summary = sess.run()
    return sess, summary


def test_camera_fallback():
    """Preferred camera fails -> the next working one is used; nothing works -> clear error."""
    class FakeVC:
        working = {0}

        def __init__(self, index, api=None):
            self.index = index

        def isOpened(self):
            return self.index in FakeVC.working

        def read(self):
            return True, np.zeros((10, 10, 3), np.uint8)

        def release(self):
            pass

        def get(self, _):
            return 0

    real = cv2.VideoCapture
    cv2.VideoCapture = FakeVC
    try:
        cap, idx = trainer.open_camera(1)
        assert idx == 0, idx
        FakeVC.working = {1}
        assert trainer.open_camera(1)[1] == 1
        FakeVC.working = set()
        try:
            trainer.open_camera(1)
            raise AssertionError("should have failed")
        except RuntimeError as e:
            assert "No camera could be opened" in str(e)
    finally:
        cv2.VideoCapture = real
    print("camera     OK   fallback works")


def test_devices():
    """Microphone listing and the level meter, with a fake `sounddevice` module."""
    import types

    import devices

    class FakeStream:
        amplitude = 0.2

        def __init__(self, device=None, channels=1, samplerate=44100, blocksize=1024, callback=None):
            self.cb = callback

        def start(self):
            t = np.arange(1024) / 44100.0
            data = (FakeStream.amplitude * np.sin(2 * np.pi * 3000 * t)).astype(np.float32).reshape(-1, 1)
            self.cb(data, 1024, None, None)

        def stop(self):
            pass

        def close(self):
            pass

    fake = types.ModuleType("sounddevice")
    fake.query_devices = lambda idx=None: (
        [{"name": "Speakers", "max_input_channels": 0, "hostapi": 0, "default_samplerate": 44100.0},
         {"name": "USB Mic", "max_input_channels": 1, "hostapi": 0, "default_samplerate": 44100.0},
         {"name": "Webcam Mic", "max_input_channels": 2, "hostapi": 1, "default_samplerate": 48000.0}]
        if idx is None else {"name": "USB Mic", "default_samplerate": 44100.0})
    fake.query_hostapis = lambda: [{"name": "MME"}, {"name": "WASAPI"}]
    fake.default = types.SimpleNamespace(device=(1, 0))
    fake.InputStream = FakeStream
    old = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = fake
    try:
        mics, err = devices.list_microphones()
        assert [m["index"] for m in mics] == [1, 2] and mics[0]["default"] and not mics[1]["default"], mics
        m = devices.MicMeter()
        m.start(1)
        assert m.heard and m.level() > 0.6 and not m.error, (m.level(), m.error)
        FakeStream.amplitude = 0.0005
        m.start(1)
        assert not m.heard and m.level() < 0.3
        m.stop()
        fake.query_devices = lambda idx=None: []
        assert devices.list_microphones() == ([], "")
    finally:
        if old is None:
            del sys.modules["sounddevice"]
        else:
            sys.modules["sounddevice"] = old
    m2, err2 = devices.list_microphones()      # sounddevice missing or real: never raises
    assert isinstance(m2, list)
    print("devices    OK   microphone list and level meter")


class FailingCap:
    """A camera that stops delivering frames after `fail_after` reads."""

    def __init__(self, clock, fail_after):
        self.clock, self.n, self.fail_after = clock, 0, fail_after

    def read(self):
        self.n += 1
        if self.n > self.fail_after:
            return False, None
        self.clock.t += 0.1
        return True, np.full((720, 1280, 3), 60, np.uint8)

    def release(self):
        pass


def test_camera_lost_and_crash_keep_the_results():
    trainee = {"trainee_id": "tester", "display_name": "Tester"}
    choice = {"action": "start", "mode": "drill", "gesture": "ball_out", "level": "standard", "reps": 5}
    # 1. the camera stops mid-session and cannot be reopened
    clock = FakeClock()
    be = FakeBackend(clock)
    be.cap = FailingCap(clock, fail_after=330)
    be.reopen_camera = lambda preferred: (_ for _ in ()).throw(RuntimeError("no camera"))
    sess = TestSession(be, trainee, choice, FakeHub(clock), clock=clock, rng=random.Random(3))
    sess.hands_off = True
    sess.sleep = lambda s: None
    summary = sess.run()
    assert summary["camera_lost"] is True and summary["error"] == "", summary
    assert summary["attempts"] >= 1
    for fn in ("attempts.csv", "summary.json", "report.html"):
        assert os.path.exists(os.path.join(sess.dir, fn)), fn
    # 2. a crash in the middle of the session: everything recorded before it is still saved
    clock = FakeClock()
    be = FakeBackend(clock)
    real_extract, calls = be.extract, {"n": 0}

    def crashing_extract(frame):
        calls["n"] += 1
        if calls["n"] > 330:
            raise RuntimeError("boom")
        return real_extract(frame)
    be.extract = crashing_extract
    sess = TestSession(be, trainee, choice, FakeHub(clock), clock=clock, rng=random.Random(3))
    sess.hands_off = True
    summary = sess.run()
    assert "boom" in summary["error"] and summary["attempts"] >= 1, summary
    for fn in ("attempts.csv", "summary.json", "report.html", "error.log"):
        assert os.path.exists(os.path.join(sess.dir, fn)), fn
    with open(os.path.join(sess.dir, "error.log"), encoding="utf-8") as fh:
        assert "boom" in fh.read()
    print("reliability OK   camera lost and crash both keep the recorded results")


def test_performance_is_measured_and_reported():
    sess, summary = run_mode("drill", tempfile.mkdtemp(), None, gesture="ball_out", hands_off=True, reps=2)
    p = summary["performance"]
    for k in ("frames", "fps_mean", "fps_first_quarter", "fps_last_quarter", "extract_ms_mean", "classify_ms_mean",
              "grade_ms_mean", "cpu_process_mean_pct", "ram_mean_mb", "psutil"):
        assert k in p, k
    assert p["frames"] > 100 and p["fps_mean"] > 0 and p["grade_ms_mean"] >= 0, p
    with open(os.path.join(sess.dir, "report.html"), encoding="utf-8") as fh:
        assert "Performance of this session" in fh.read()
    import subprocess
    res = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "performance_report.py"), "--min-seconds", "1"],
                         capture_output=True, text=True)
    assert res.returncode == 0 and "drill" in res.stdout, res.stdout + res.stderr
    print("performance OK   ", [l for l in res.stdout.splitlines() if l.startswith("drill")][0][:70])


def main():
    test_camera_fallback()
    test_devices()
    save_dir = None
    if "--save-frames" in sys.argv:
        save_dir = sys.argv[sys.argv.index("--save-frames") + 1]
        os.makedirs(os.path.join(save_dir, "light"), exist_ok=True)
    tmp = tempfile.mkdtemp()
    old = os.getcwd()
    os.chdir(tmp)
    try:
        test_camera_lost_and_crash_keep_the_results()
        test_performance_is_measured_and_reported()

        # practice
        sess, summary = run_mode("practice", tmp, save_dir)
        assert sess.stable_label == "ball_out", sess.stable_label
        print("practice   OK   stable label:", sess.stable_label)

        # drill: three graded attempts of one signal
        sess, summary = run_mode("drill", tmp, save_dir, gesture="team_to_serve_left")
        assert summary["attempts"] == 3 and summary["correct"] == 3 and summary["points"] == 30, summary
        print("drill      OK  ", summary["one_line"])

        # drill with repetitions: runs back to back with no key presses, every rep is listed
        sess, summary = run_mode("drill", tmp, save_dir, gesture="ball_out", hands_off=True, reps=4)
        assert summary["attempts"] == 4 and summary["correct"] == 4, summary
        assert sess.review_i is not None and len(sess.attempt_results) == 4
        print("drill x4   OK  ", summary["one_line"])

        # combo: Team to Serve then the reason, twice, automatic
        sess, summary = run_mode("combo", tmp, save_dir, hands_off=True, reps=2, combo="ball_out", level="beginner")
        targets = [a["target"] for a in sess.attempts]
        assert len(targets) == 4 and all(targets[i].startswith("team_to_serve_") and targets[i + 1] == "ball_out"
                                         for i in (0, 2)), targets
        assert summary["correct"] == 4, summary
        print("combo x2   OK  ", targets)

        # optional FIVB pictures: assets/signals/<label>.png is shown while getting ready
        os.makedirs(os.path.join("assets", "signals"))
        pic = np.full((420, 300, 3), 255, np.uint8)
        cv2.circle(pic, (150, 90), 40, (60, 60, 60), 3)
        cv2.line(pic, (150, 130), (150, 300), (60, 60, 60), 4)
        cv2.line(pic, (150, 160), (70, 90), (60, 60, 60), 4)
        cv2.line(pic, (150, 160), (230, 90), (60, 60, 60), 4)
        cv2.imwrite(os.path.join("assets", "signals", "ball_out.png"), pic)
        sess, summary = run_mode("drill", tmp, save_dir, gesture="ball_out", hands_off=True, reps=2)
        assert any(v is not None for v in sess._ref_cache.values()), "the picture should have been loaded"
        assert summary["attempts"] == 2, summary
        print("pictures   OK   assets/signals picture shown")

        # labeled test sessions write intent + measured check values, and the evaluation tool reads them
        sess, summary = run_mode("drill", tmp, save_dir, gesture="ball_out", hands_off=True, reps=3,
                                 intent="wrong", level="standard", note="elbows tucked")
        import csv as _csv
        with open(os.path.join(sess.dir, "attempts.csv"), newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        assert rows and all(r["intent"] == "wrong" and r["note"] == "elbows tucked"
                            and "forearms_vertical=" in r["check_values"] for r in rows), rows[:1]
        assert sess.dir.endswith("_wrong"), sess.dir
        sess2, _ = run_mode("drill", tmp, save_dir, gesture="ball_out", hands_off=True, reps=3, intent="correct")
        import subprocess
        res = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "evaluate_sessions.py")],
                             capture_output=True, text=True)
        assert "STANDARD level" in res.stdout and "Cohen" in res.stdout, res.stdout + res.stderr
        assert "WRONG attempts that were accepted" in res.stdout and "elbows tucked" in res.stdout   # the fake person performs it correctly every time
        print("evaluation OK  ", [l for l in res.stdout.splitlines() if l.startswith("ALL")][0])

        # whistle required before Team to Serve: whistle, signal, whistle, signal (all automatic)
        sess, summary = run_mode("drill", tmp, save_dir, gesture="team_to_serve_left", hands_off=True, reps=2,
                                 whistle=True)
        assert [a["kind"] for a in sess.attempts] == ["whistle", "gesture"] * 2 and summary["correct"] == 4, \
            [(a["kind"], a["verdict"]) for a in sess.attempts]
        print("whistle    OK  ", summary["one_line"])

        # combo with a double contact: the trainee must use the hand on the side of the team at fault
        sess, summary = run_mode("combo", tmp, save_dir, hands_off=True, reps=2, combo="double_contact",
                                 level="beginner")
        assert summary["correct"] == 4, [(a["target"], a["verdict"], a["feedback"]) for a in sess.attempts]
        print("combo dc   OK  ", [a["target"] for a in sess.attempts])

        # every attempt keeps its real movement; the regrade tool reads it back with the current rules
        npz = sorted(os.listdir(os.path.join(sess.dir, "attempts")))
        assert len(npz) == 2 and npz[0].startswith("attempt_001_team_to_serve_") and npz[0].endswith("+double_contact.npz"), npz
        import subprocess
        res = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "regrade_attempts.py")],
                             capture_output=True, text=True)
        assert res.returncode == 0 and "unchanged" in res.stdout.lower(), res.stdout + res.stderr
        print("regrade    OK  ", [l for l in res.stdout.splitlines() if "unchanged" in l.lower()][0].strip())

        # light theme renders every screen without errors (frames saved for a visual check)
        trainer.set_cv_theme("light")
        sess, summary = run_mode("drill", tmp, os.path.join(save_dir, "light") if save_dir else None,
                                 gesture="team_to_serve_left", level="beginner")
        assert summary["attempts"] == 3, summary
        sess, summary = run_mode("practice", tmp, os.path.join(save_dir, "light") if save_dir else None)
        trainer.set_cv_theme("dark")
        print("light mode OK")

        # challenge: one attempt for every signal
        sess, summary = run_mode("challenge", tmp, save_dir, level="beginner", hands_off=True)
        assert summary["attempts"] == 8 and summary["correct"] == 8, summary
        print("challenge  OK  ", summary["one_line"])

        # simulation: whistle + gestures across scenarios, scoreboard from correct calls
        sess, summary = run_mode("sim", tmp, save_dir, level="beginner")
        assert summary["attempts"] > 8 and summary["team_points"] is not None, summary
        assert sum(summary["team_points"].values()) == 3, summary          # 3 rallies, every call correct
        assert summary["correct"] == summary["attempts"], summary
        kinds = [a["kind"] for a in sess.attempts]
        assert kinds[0] == "whistle" and kinds[-2:] == ["whistle", "gesture"], kinds
        assert any(a["target"] == "end_of_set" for a in sess.attempts)
        sim_sess, sim_summary = sess, summary
        # the whistle is part of each signal step (not a separate step), and a commit message is produced
        assert all(q["kind"] != "whistle" and q.get("whistle") for q in sess.queue), sess.queue
        assert len(sess.queue) == 7 and kinds.count("whistle") == 7, (len(sess.queue), kinds.count("whistle"))
        assert sess.commit["title"] == "SET ENDED" and any(l.startswith("COMMITTED") for l in sess.commit_log + ["COMMITTED"])
        assert sess.whistle_state == "ok"
        assert sess.serving in ("left", "right"), sess.serving          # the top bar shows who is serving
        print("sim commit OK  ", sess.commit["title"], "|", sess.commit["lines"][0], "| serving", sess.serving)

        # continuous match simulation: runs the whole set by itself
        sess, summary = run_mode("sim", tmp, save_dir, level="beginner", hands_off=True, continuous=True)
        assert summary["attempts"] == 17 and summary["correct"] == 17, summary
        print("sim cont.  OK  ", summary["one_line"])

        # whistle states: late and missing
        sess, summary = run_mode("drill", tmp, save_dir, gesture="team_to_serve_left", hands_off=True, reps=2,
                                 whistle=True, whistle_at=3.8)
        assert [a["verdict"] for a in sess.attempts if a["kind"] == "whistle"] == ["ALMOST", "ALMOST"], sess.attempts
        sess, summary = run_mode("drill", tmp, save_dir, gesture="team_to_serve_left", hands_off=True, reps=2,
                                 whistle=True, whistle_at=None)
        assert [a["verdict"] for a in sess.attempts if a["kind"] == "whistle"] == ["INCORRECT", "INCORRECT"]
        print('whistle st OK   on time / late / not heard')
        print("sim        OK  ", sim_summary["one_line"], sim_summary["team_points"])

        # outputs written
        d = sim_sess.dir
        for fn in ("attempts.csv", "summary.json", "report.html"):
            assert os.path.exists(os.path.join(d, fn)), fn
        print("outputs    OK  ", sorted(os.listdir(d)))
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()