"""
skeleton_check.py (v2 — multi-target, back/forward navigation, verdict logging)

Visual review pass for spot-checking whether clips are MISLABELED —
i.e. whether the referee's actual motion in the video matches the
gesture class/folder it's filed under. This is a content check, not
a detection-quality check (that's what clip_reviewer.py's flag/keep
is more about, though it can surface mislabeling too).

Shows original + skeleton overlay side by side, exactly like
clip_reviewer.py, but is meant to move fast through a TARGETED list
of (gesture, person) combos instead of one full class at a time.

--------------------------------------------------------------------
CHANGES FROM v1:
- TARGETS is now a list of (gesture, person) pairs instead of a single
  hardcoded GESTURE/PERSON. Defaults to the 4 classes/people flagged
  as suspect: ball_out, double_contact, end_of_set,
  service_authorization_left, each for p08 and p09.
- Full back/forward navigation across the ENTIRE flattened clip list
  (not just within one gesture+person combo) — B goes back even
  across a class boundary, N/SPACE goes forward.
- Verdict logging: press C (correct — matches its label), M
  (mislabeled — does NOT match its label), U (unsure), or just N to
  skip without a verdict. Saved to data/skeleton_check_progress.json,
  resumable across runs, same pattern as clip_reviewer.py's progress
  file.
- The label is printed BIG on screen (not just in the console) since
  content-checking means you need to keep the expected label in mind
  while watching, not just read it once at the start.
--------------------------------------------------------------------

Controls (while a clip is playing):
  SPACE      - pause / resume playback
  C          - verdict: CORRECT (matches its label), move to next
  M          - verdict: MISLABELED, move to next
  U          - verdict: UNSURE / needs a second look, move to next
  N          - skip without a verdict, move to next
  B          - go back to the previous clip
  R          - restart current clip from the beginning
  =  / -     - speed up / slow down playback (cycles through SPEED_LEVELS)
  Q          - quit and save progress

SPEED CONTROL:
At higher speeds (2x+), full MediaPipe pose+hand inference on every
single frame becomes the bottleneck, not the video decode or display.
So above SPEED_SKIP_THRESHOLD, the skeleton overlay is only recomputed
every Nth frame (N scales with speed) and reused for the frames in
between -- the ORIGINAL video (left panel) still plays every frame at
the higher speed, only the right-panel skeleton updates less often.
This keeps iteration fast while still giving you enough skeleton
context to judge gesture identity, since a gesture's shape barely
changes frame-to-frame anyway.

Run from your project root:
    python skeleton_check.py
"""

import os
import csv
import json
import cv2
import numpy as np
import mediapipe as mp

from extract_keypoints import extract_pose_features, extract_hand_features, debug_get_crop_info

MANIFEST_PATH = "data/dataset_manifest.csv"
PROGRESS_PATH = "data/skeleton_check_progress.json"
DISPLAY_HEIGHT = 480
WINDOW_NAME = "Skeleton Check (mislabel review)"
MAX_READ_FAILURES = 30  # same hang-prevention fix as clip_reviewer.py

# Playback speed control. BASE_DELAY_MS is the real-time-ish delay at 1x.
# Actual delay = max(1, int(BASE_DELAY_MS / speed)).
SPEED_LEVELS = [0.5, 1, 2, 4, 8]
DEFAULT_SPEED_INDEX = 2  # starts at 2x -- iterating fast is the whole point here
BASE_DELAY_MS = 30
# Above this speed, skip MediaPipe inference on most frames and reuse the
# last computed skeleton overlay instead of recomputing every frame.
SPEED_SKIP_THRESHOLD = 2

# The suspect classes/people from the Dwayne/Liam regression, in the
# order you want to work through them. Edit freely -- add/remove
# (gesture, person) pairs as needed.
TARGETS = [
    ("ball_out", "p08"),
    ("ball_out", "p09"),
    ("double_contact", "p08"),
    ("double_contact", "p09"),
    ("end_of_set", "p08"),
    ("end_of_set", "p09"),
    ("service_authorization_left", "p08"),
    ("service_authorization_left", "p09"),
]

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_manifest_rows():
    with open(MANIFEST_PATH, "r") as f:
        return list(csv.DictReader(f))


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"verdicts": {}, "last_index": 0}


def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def build_flat_clip_list(all_rows):
    """
    Builds one flat, ordered list of rows covering every (gesture, person)
    pair in TARGETS, so B/N navigation can move across class boundaries
    seamlessly instead of being trapped inside one combo.
    """
    flat = []
    for gesture, person in TARGETS:
        matching = [r for r in all_rows if r["gesture_label"] == gesture and r["person_id"] == person]
        if not matching:
            print(f"  (no clips found for {gesture} / {person} -- skipping)")
            continue
        flat.extend(matching)
    return flat


def draw_skeleton_frame(frame_bgr, pose_model, hands_model):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose_model.process(frame_rgb)
    pose_features, pose_landmarks = extract_pose_features(pose_results)
    hand_coords, left_det, right_det, _, _ = extract_hand_features(frame_rgb, pose_landmarks, hands_model)

    skeleton_frame = frame_bgr.copy()
    h, w = skeleton_frame.shape[:2]

    if pose_results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            skeleton_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )

    left_points = hand_coords[:42].reshape(21, 2)
    right_points = hand_coords[42:].reshape(21, 2)
    if left_det > 0.5:
        for x, y in left_points:
            cv2.circle(skeleton_frame, (int(x * w), int(y * h)), 3, (255, 255, 0), -1)
    if right_det > 0.5:
        for x, y in right_points:
            cv2.circle(skeleton_frame, (int(x * w), int(y * h)), 3, (255, 0, 255), -1)

    mode, boxes = debug_get_crop_info(pose_landmarks, w, h)
    box_color = (0, 165, 255) if mode == "combined" else (255, 100, 0)
    for box in boxes:
        cv2.rectangle(skeleton_frame, box[:2], box[2:], box_color, 2)

    return skeleton_frame


def resize_to_height(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / h
    return cv2.resize(frame, (int(w * scale), target_height))


def main():
    all_rows = load_manifest_rows()
    if not all_rows:
        print("Manifest is empty or missing. Run build_manifest.py first.")
        return

    flat_rows = build_flat_clip_list(all_rows)
    if not flat_rows:
        print("No clips found for any TARGETS combo. Check the TARGETS list / manifest.")
        return

    progress = load_progress()
    index = progress.get("last_index", 0)
    index = max(0, min(index, len(flat_rows) - 1))

    pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=0.1, min_tracking_confidence=0.28)

    speed_index = DEFAULT_SPEED_INDEX  # persists across clips within this run, not reset per clip

    print(f"Reviewing {len(flat_rows)} clips across {len(TARGETS)} (gesture, person) targets.")
    print(f"Resuming at clip {index + 1}/{len(flat_rows)}.")
    print(f"Starting speed: {SPEED_LEVELS[speed_index]}x")
    print("SPACE=pause/resume  C=correct  M=mislabeled  U=unsure  N=skip  B=back  R=restart  +/-=speed  Q=quit\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 680)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    quit_all = False
    while 0 <= index < len(flat_rows) and not quit_all:
        row = flat_rows[index]
        clip_path = row["clip_path"]
        gesture_label = row["gesture_label"]
        person_id = row["person_id"]
        prior_verdict = progress["verdicts"].get(clip_path, "unreviewed")

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print(f"  Could not open {clip_path} -- skipping.")
            index += 1
            continue

        paused = False
        advance_action = None
        consecutive_read_failures = 0
        frame_counter = 0
        cached_skeleton_frame = None  # reused on skipped frames at high speed

        while advance_action is None:
            speed = SPEED_LEVELS[speed_index]
            delay_ms = max(1, int(BASE_DELAY_MS / speed))
            # Recompute skeleton every frame at low speed; skip more frames
            # as speed increases, since MediaPipe inference is the real
            # bottleneck, not video decode/display.
            skeleton_every_n = 1 if speed < SPEED_SKIP_THRESHOLD else int(speed)

            if not paused:
                success, frame = cap.read()
                if not success:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= MAX_READ_FAILURES:
                        print(f"  WARNING: {clip_path} won't decode -- skipping.")
                        advance_action = "next"
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                consecutive_read_failures = 0

                if frame_counter % skeleton_every_n == 0 or cached_skeleton_frame is None:
                    skeleton_frame = draw_skeleton_frame(frame, pose_model, hands_model)
                    cached_skeleton_frame = skeleton_frame
                else:
                    # Reuse last computed overlay -- gesture shape barely
                    # changes frame-to-frame, so this stays visually useful
                    # while skipping the expensive inference call.
                    skeleton_frame = cached_skeleton_frame
                frame_counter += 1

                left_panel = resize_to_height(frame, DISPLAY_HEIGHT)
                right_panel = resize_to_height(skeleton_frame, DISPLAY_HEIGHT)
                combined = np.hstack([left_panel, right_panel])

                # Expected label shown BIG -- this is a content check, keep
                # the label in view the whole time you're watching.
                cv2.putText(combined, f"LABEL: {gesture_label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(combined, f"person: {person_id}  |  [{index+1}/{len(flat_rows)}]  |  prior verdict: {prior_verdict}  |  speed: {speed}x",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(combined, os.path.basename(clip_path), (10, DISPLAY_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                controls_lines = [
                    "SPACE=pause  C=correct  M=MISLABELED  U=unsure",
                    "N=skip  B=back  R=restart  +/- =speed  Q=quit",
                ]
                for i, line in enumerate(controls_lines):
                    cv2.putText(combined, line, (10, DISPLAY_HEIGHT - 65 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF

            if key == ord(' '):
                paused = not paused
            elif key == ord('c'):
                progress["verdicts"][clip_path] = "correct"
                advance_action = "next"
            elif key == ord('m'):
                progress["verdicts"][clip_path] = "mislabeled"
                print(f"  -> flagged MISLABELED: {clip_path}")
                advance_action = "next"
            elif key == ord('u'):
                progress["verdicts"][clip_path] = "unsure"
                advance_action = "next"
            elif key == ord('n'):
                advance_action = "next"
            elif key == ord('b'):
                advance_action = "prev"
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                consecutive_read_failures = 0
                frame_counter = 0
            elif key in (ord('='), ord('+')):
                speed_index = min(len(SPEED_LEVELS) - 1, speed_index + 1)
            elif key == ord('-'):
                speed_index = max(0, speed_index - 1)
            elif key == ord('q'):
                advance_action = "quit"
                quit_all = True

        cap.release()

        if advance_action == "prev":
            index = max(0, index - 1)
        else:
            index += 1

        progress["last_index"] = index
        save_progress(progress)

    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()

    mislabeled = [k for k, v in progress["verdicts"].items() if v == "mislabeled"]
    unsure = [k for k, v in progress["verdicts"].items() if v == "unsure"]
    correct_count = sum(1 for v in progress["verdicts"].values() if v == "correct")

    print(f"\nSession summary: {correct_count} correct, {len(mislabeled)} mislabeled, {len(unsure)} unsure.")
    if mislabeled:
        print("\nMISLABELED clips flagged this pass:")
        for path in mislabeled:
            print(f"  - {path}")
    if unsure:
        print("\nUNSURE clips (worth a second look):")
        for path in unsure:
            print(f"  - {path}")


if __name__ == "__main__":
    main()