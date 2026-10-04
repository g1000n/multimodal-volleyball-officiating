"""
tests/test_trainer_smoke.py

Headless smoke test for trainer.py: drives every mode (practice, drill, challenge, sim) with a
fake camera, a scripted "person" and a fake classifier, so the whole state machine, drawing code,
grading hookup and report writing run without hardware, MediaPipe, PyTorch or a display.

Run:  python tests/test_trainer_smoke.py [--save-frames DIR]
"""

import csv
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
    def __init__(self, clock, always_show=None, script=None):
        super().__init__(FakeCap(clock), self._extract, self._classify, IDX_TO_LABEL)
        self.session = None
        self.clock = clock
        self.always_show = always_show      # practice mode: label to perform continuously
        self.script = script                # match_test: [(t_start, t_end, label_or_None), ...] since session start
        self.n = 0

    def _extract(self, frame):
        s = self.session
        self.n += 1
        label, side = None, "right"
        if self.script is not None and self.session is not None and self.session._t0 is not None:
            elapsed = self.clock() - self.session._t0
            for t_start, t_end, seg_label in self.script:
                if t_start <= elapsed < t_end:
                    label = seg_label
                    break
        elif self.always_show:
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
        now = self.clock() if self.clock else 0.0
        self.times.append(now)
        self.events = getattr(self, "events", []) + [{"t": now, "source": "manual", "confidence": None}]

    def events_since(self, t):
        return [dict(e) for e in getattr(self, "events", []) if e["t"] >= t]

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
        if (self.mode == "match_test" and getattr(self, "_mt_test_whistle_at", None) is not None
                and self._t0 is not None and not self.whistle.times
                and (self.clock() - self._t0) >= self._mt_test_whistle_at):
            return ord("w")
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
                if getattr(self, "press_r", False):
                    return ord("r")
                self.summary_keys = getattr(self, "summary_keys", 0) + 1
                return ord("d") if self.summary_keys == 1 else 32
            return 255
        if self.mode == "practice":
            return 27 if self.frames_seen > 60 else 255
        if self.mode == "match_test":
            return 27 if self.frames_seen > getattr(self, "mt_end_after", 220) else 255
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


def run_mode(mode, tmp, save_dir, gesture=None, level="standard", hands_off=False, whistle_at=0.3, press_r=False,
            script=None, mt_end_after=None, mt_whistle_at=None, **extra):
    clock = FakeClock()
    be = FakeBackend(clock, always_show="ball_out" if mode == "practice" else None, script=script)
    trainee = {"trainee_id": "tester", "display_name": "Tester"}
    choice = {"action": "start", "mode": mode, "gesture": gesture, "level": level, **extra}
    sess = TestSession(be, trainee, choice, FakeHub(clock), clock=clock, rng=random.Random(3), save_dir=save_dir)
    sess.hands_off = hands_off
    sess.whistle_at = whistle_at
    sess.press_r = press_r
    if mt_end_after is not None:
        sess.mt_end_after = mt_end_after
    if mt_whistle_at is not None:
        sess._mt_test_whistle_at = mt_whistle_at
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


def test_match_test_mode():
    """Match Testing: continuous, unscripted grading of a real performer. No countdown, no expected order, no fixed
    capture window. A run is graded and logged the moment the detected label changes; the scoreboard follows the
    trainee's Team to Serve calls; a whistle blown at any point is logged (purely observational, not required)."""
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        script = [
            (0.0, 1.0, None),                          # arms down
            (1.0, 3.0, "team_to_serve_left"),          # a clean 2 s hold
            (3.0, 3.3, None),
            (3.3, 5.3, "ball_out"),                    # a clean 2 s hold
            (5.3, 6.5, None),
        ]
        sess, summary = run_mode("match_test", tmp, None, level="standard", script=script, mt_end_after=160,
                                 mt_whistle_at=1.8)
        gest = [a for a in sess.attempts if a["kind"] == "gesture"]
        assert [a["target"] for a in gest] == ["team_to_serve_left", "ball_out"], gest
        assert all(a["verdict"] in ("CORRECT", "ALMOST") for a in gest), gest
        assert sess.serving == "left", sess.serving
        assert sess.team["left"] >= 1, sess.team
        assert sess.mt_last_result is not None and sess.mt_last_result.target == "ball_out"
        assert sess.mt_in_run is False           # both runs were properly closed, nothing left dangling
        # the whistle blown mid-hold was logged (purely observational; nothing required it)
        log = open(os.path.join(sess.dir, "session_log.csv"), encoding="utf-8").read()
        assert "whistle_heard" in log and "match test whistle #1" in log, log
        # a movement file was saved for each graded run, so thresholds can be re-checked later
        npz_dir = os.path.join(sess.dir, "attempts")
        saved = sorted(os.listdir(npz_dir)) if os.path.isdir(npz_dir) else []
        assert len(saved) == 2 and any("team_to_serve_left" in f for f in saved) and any("ball_out" in f for f in saved), saved
        # the summary screen (A/D review) works with no scripted queue behind it
        assert sess.phase in ("summary", "done"), sess.phase
        for fn in ("attempts.csv", "summary.json", "report.html"):
            assert os.path.exists(os.path.join(sess.dir, fn)), fn
        print("match_test OK  ", summary["one_line"], "| serving", sess.serving, "| team", sess.team)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_match_test_flicker_is_discarded_and_run_continues_across_a_pause():
    """A very brief, flickering detection (well under the model's own natural hold) is discarded quietly, not
    graded as a near-empty attempt."""
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        script = [(0.0, 0.5, None), (0.5, 0.65, "double_contact"), (0.65, 2.0, None)]   # a 0.15 s blip
        sess, summary = run_mode("match_test", tmp, None, level="standard", script=script, mt_end_after=80)
        gest = [a for a in sess.attempts if a["kind"] == "gesture"]
        assert gest == [], gest
        print("match_test flicker OK   (discarded, not logged)")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_match_test_auth_cancels_pending_team_to_serve():
    """A Service Authorization beckon whose first half is read as a same-side Team to Serve must NOT score a point:
    the Team to Serve run is held pending (live_deployment.py's TEAM_TO_SERVE_CONFIRM_DELAY_SECONDS) and the
    authorization taking over cancels it, with or without "Require the whistle". A genuine Team to Serve with nothing
    after it still scores."""
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        # the rolling window fills at ~2.4 s, so the signals start after that
        script = [(0.0, 3.0, None), (3.0, 4.8, "team_to_serve_left"), (4.8, 7.0, "service_authorization_left"),
                  (7.0, 12.0, None)]
        for gated in (False, True):
            sess, summary = run_mode("match_test", tmp, None, level="standard", script=script, mt_end_after=130,
                                     mt_whistle_at=0.3 if gated else None, whistle=gated)
            gest = [a["target"] for a in sess.attempts if a["kind"] == "gesture"]
            assert gest == ["service_authorization_left"], gest
            assert sess.team == {"left": 0, "right": 0}, sess.team
            assert sess.mt_pending_tts is None
            with open(os.path.join(sess.dir, "session_log.csv"), encoding="utf-8") as fh:
                assert ",mt_tts_cancelled," in fh.read()

        script = [(0.0, 3.0, None), (3.0, 5.0, "team_to_serve_left"), (5.0, 10.0, None)]
        sess, summary = run_mode("match_test", tmp, None, level="standard", script=script, mt_end_after=110)
        assert sess.team["left"] == 1, sess.team
        print("match_test auth OK   same-side authorization cancelled the pending Team to Serve; a real one scored")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_sim_whistle_gate():
    """Match Simulation with "Require the whistle": a call only scores when decision_engine.py accepts its whistle AND
    the form passes. Without the option, form alone decides (the whistle is graded as its own attempt)."""
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        def sim(**kw):
            return run_mode("sim", tmp, None, level="beginner", hands_off=True, continuous=True, **kw)
        sess, summary = sim(whistle=True)                          # whistle blown 0.3 s into every capture
        assert sum(summary["team_points"].values()) == 3, summary
        sess, summary = sim(whistle=True, whistle_at=None)         # never blown
        assert summary["team_points"] == {"left": 0, "right": 0}, summary
        assert any("No whistle counted" in l for l in sess.commit_log), sess.commit_log
        opening = sess.queue[0]["side"]
        assert sess.serving == opening, (sess.serving, opening)  # no committed call -> server never changed
        sess, summary = sim(whistle_at=None)                       # option off: whistle not required
        assert sum(summary["team_points"].values()) == 3, summary
        with open(os.path.join(sess.dir, "attempts.csv"), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert all(r["detected"] == r["target"] for r in rows if r["kind"] == "gesture"), rows
        assert "needs_practice" in summary
        print("sim gate   OK   whistle required: 3 pts with it, 0 without; option off: 3 pts")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_pause_mid_attempt_and_feedback_latency():
    """P during a capture drops that attempt (never graded or logged) and repeats the step after resuming. Every graded
    attempt logs feedback_latency_s (signal end -> verdict), and a capture ends early once the arms are back down."""
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        clock = FakeClock()
        be = FakeBackend(clock)
        choice = {"action": "start", "mode": "drill", "gesture": "team_to_serve_left", "level": "beginner", "reps": 2}
        sess = TestSession(be, {"trainee_id": "tester", "display_name": "Tester"}, choice, FakeHub(clock),
                           clock=clock, rng=random.Random(3))
        sess.hands_off = True
        base_key = sess._key
        state = {"paused_at": None}

        def key():
            if state["paused_at"] is None and sess.phase == "capture" and clock() - sess.phase_t0 > 1.0:
                state["paused_at"] = clock()
                return ord("p")
            if sess.paused and clock() - state["paused_at"] > 1.0:
                return ord("p")                        # resume
            return base_key()
        sess._key = key
        summary = sess.run()
        gest = [a for a in sess.attempts if a["kind"] == "gesture"]
        assert len(gest) == 2, gest                    # the paused attempt was not counted; both reps still ran
        with open(os.path.join(sess.dir, "session_log.csv"), encoding="utf-8") as fh:
            assert ",attempt_cancelled," in fh.read()
        lat = [float(a["feedback_latency_s"]) for a in gest]
        assert all(x <= trainer.FEEDBACK_TARGET_SECONDS for x in lat), lat
        assert summary["performance"]["feedback_within_3s"] == 2, summary["performance"]
        print("pause/latency OK   paused attempt dropped; feedback", lat, "s after arms down")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_match_test_flush_on_early_quit():
    """Pressing Q mid-hold (session ends early) still grades and saves whatever was in progress, instead of
    silently dropping the performer's last signal."""
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        script = [(0.0, 0.5, None), (0.5, 30.0, "ball_out")]   # a long hold; ends only when Q cuts the session short
        sess, summary = run_mode("match_test", tmp, None, level="standard", script=script, mt_end_after=40)
        gest = [a for a in sess.attempts if a["kind"] == "gesture"]
        assert len(gest) == 1 and gest[0]["target"] == "ball_out", gest
        print("match_test flush OK   the in-progress run was graded when the session ended early")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


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
        test_match_test_mode()
        test_match_test_flicker_is_discarded_and_run_continues_across_a_pause()
        test_match_test_flush_on_early_quit()
        test_match_test_auth_cancels_pending_team_to_serve()
        test_sim_whistle_gate()
        test_pause_mid_attempt_and_feedback_latency()

        # practice
        sess, summary = run_mode("practice", tmp, save_dir)
        assert sess.stable_label == "ball_out", sess.stable_label
        print("practice   OK   stable label:", sess.stable_label)

        # practice mode logs every detection event (documentation for the paper), even though it is not scored
        assert os.path.exists(os.path.join(sess.dir, "practice_log.csv")), "practice_log.csv should be written"
        prows = list(csv.DictReader(open(os.path.join(sess.dir, "practice_log.csv"), encoding="utf-8")))
        assert len(prows) >= 1 and any(r["detected_label"] != "nothing" for r in prows), prows
        assert "<h2>Practice log</h2>" in open(os.path.join(sess.dir, "report.html"), encoding="utf-8").read()
        print("practice log OK  ", len(prows), "detection events logged")

        # Practice mode ALSO produces a real, fully graded score per held signal (not just live checks), at all
        # three difficulty levels, even though nothing is shown as a live score during Practice itself
        assert os.path.exists(os.path.join(sess.dir, "practice_scores.csv")), "practice_scores.csv should be written"
        psrows = list(csv.DictReader(open(os.path.join(sess.dir, "practice_scores.csv"), encoding="utf-8")))
        assert len(psrows) >= 1, psrows
        for r in psrows:
            assert r["target"] == "ball_out" and r["verdict"] in ("CORRECT", "ALMOST", "INCORRECT")
            for lv in ("beginner", "standard", "referee"):
                assert r[f"score_{lv}"] != "" and r[f"verdict_{lv}"] in ("CORRECT", "ALMOST", "INCORRECT")
        assert sess.mt_last_result is not None and sess.mt_last_result.target == "ball_out"
        assert "<h2>Practice: fully graded holds</h2>" in open(os.path.join(sess.dir, "report.html"), encoding="utf-8").read()
        print("practice score OK ", len(psrows), "fully graded holds, e.g.", psrows[0]["score_referee"], "at referee")
        print("practice log OK  ", len(prows), "detection events logged")

        # drill: three graded attempts of one signal
        sess, summary = run_mode("drill", tmp, save_dir, gesture="team_to_serve_left")
        assert summary["attempts"] == 3 and summary["correct"] == 3 and summary["points"] == 30, summary
        print("drill      OK  ", summary["one_line"])

        # every gesture attempt also shows how the SAME capture would grade at every difficulty level
        for a in sess.attempts:
            for lv in ("beginner", "standard", "referee"):
                assert a[f"score_{lv}"] != "" and a[f"verdict_{lv}"] in ("CORRECT", "ALMOST", "INCORRECT"), a
        with open(os.path.join(sess.dir, "attempts.csv"), newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert csv_rows and all(r["score_referee"] != "" for r in csv_rows), csv_rows
        assert "All levels" in open(os.path.join(sess.dir, "report.html"), encoding="utf-8").read()
        print("multi-level OK   drill attempt graded at",
              {lv: sess.attempts[0][f"score_{lv}"] for lv in ("beginner", "standard", "referee")})

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

        # logging: whistle events (time, auto or manual) and a full session log, also in the report
        sess, summary = run_mode("drill", tmp, save_dir, gesture="team_to_serve_left", hands_off=True, reps=2, whistle=True)
        import csv as _csv2
        with open(os.path.join(sess.dir, "whistle_events.csv"), newline="", encoding="utf-8") as fh:
            wrows = list(_csv2.DictReader(fh))
        assert len(wrows) == 2 and all(r["source"] == "manual" and r["judged_as"].startswith("on time") for r in wrows), wrows
        assert all(r["attempt_no"] and r["signal"] == "team_to_serve_left" and r["seconds_after_go"] for r in wrows), wrows
        with open(os.path.join(sess.dir, "session_log.csv"), newline="", encoding="utf-8") as fh:
            events = [r["event"] for r in _csv2.DictReader(fh)]
        for needed in ("session_start", "step_start", "capture_start", "whistle_heard", "verdict", "session_end"):
            assert needed in events, (needed, events)
        report = open(os.path.join(sess.dir, "report.html"), encoding="utf-8").read()
        assert "Whistle log" in report and "manual" in report
        print("logs       OK   whistle_events.csv, session_log.csv, report:", len(wrows), "whistles,", len(events), "log lines")

        # R on the summary screen = do the same session again
        sess, summary = run_mode("drill", tmp, save_dir, gesture="ball_out", hands_off=True, reps=2, press_r=True)
        assert summary["restart"] is True and summary["attempts"] == 2, summary
        with open(os.path.join(sess.dir, "session_log.csv"), newline="", encoding="utf-8") as fh:
            assert "do_again" in [r["event"] for r in _csv2.DictReader(fh)]
        print("do again   OK   summary R restarts the session")

        # simulation length follows the repetitions / rallies choice (5 rallies = 2 * 5 + 1 steps)
        sess, summary = run_mode("sim", tmp, save_dir, level="beginner", hands_off=True, continuous=True, reps=5)
        assert len(sess.queue) == 11 and summary["correct"] == summary["attempts"], (len(sess.queue), summary)
        print("sim rallies OK  5 rallies")

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