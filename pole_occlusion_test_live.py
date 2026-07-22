"""
pole_occlusion_test_live.py

Same idea as pole_occlusion_test.py, but LIVE — uses your webcam/iPhone
feed (via Camo, same as live_auto_inference.py) instead of a pre-recorded
clip. Lets you physically stand in front of the camera, adjust a
simulated occluding rectangle (like a net pole) over yourself in real
time using sliders, and watch whether pose/hand detection survives.

Run:
    python pole_occlusion_test_live.py

Controls:
  Q - quit
  (sliders in the "Controls" window adjust the rectangle's X, Y,
   Width, Height live)
"""

import cv2
import numpy as np
import mediapipe as mp

from extract_keypoints import extract_pose_features, extract_hand_features

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

CAMERA_INDEX = 1  # set to whatever index your iPhone/Camo feed showed in list_cameras.py
DISPLAY_HEIGHT = 480
OCCLUSION_COLOR = (15, 15, 15)  # near-black, dark like a solid pole


def draw_skeleton_frame(frame_bgr, pose_model, hands_model):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose_model.process(frame_rgb)
    pose_features, pose_landmarks = extract_pose_features(pose_results)
    hand_coords, left_det, right_det, _, _ = extract_hand_features(frame_rgb, pose_landmarks, hands_model)

    skeleton_frame = frame_bgr.copy()
    frame_height, frame_width = skeleton_frame.shape[:2]

    if pose_results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            skeleton_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )
    else:
        cv2.putText(skeleton_frame, "POSE LOST", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    left_points = hand_coords[:42].reshape(21, 2)
    right_points = hand_coords[42:].reshape(21, 2)

    if left_det > 0.5:
        for x, y in left_points:
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(skeleton_frame, (px, py), 3, (255, 255, 0), -1)

    if right_det > 0.5:
        for x, y in right_points:
            px, py = int(x * frame_width), int(y * frame_height)
            cv2.circle(skeleton_frame, (px, py), 3, (255, 0, 255), -1)

    return skeleton_frame


def resize_to_height(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / h
    return cv2.resize(frame, (int(w * scale), target_height))


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}.")
        return

    success, first_frame = cap.read()
    if not success:
        print("Could not read a frame from the camera.")
        return

    frame_h, frame_w = first_frame.shape[:2]

    pose_model = mp_pose.Pose(
        static_image_mode=False, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    hands_model = mp_hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=0.1, min_tracking_confidence=0.28,
    )

    controls_window = "Controls"
    display_window = "Pole Occlusion Test (Live)"

    cv2.namedWindow(controls_window)
    cv2.resizeWindow(controls_window, 400, 200)
    # Default rectangle: a vertical strip roughly in the middle, like a
    # pole would look from a straight-on camera angle. Adjust freely.
    default_w = max(10, frame_w // 15)
    default_h = frame_h
    cv2.createTrackbar("X", controls_window, frame_w // 2 - default_w // 2, frame_w, nothing)
    cv2.createTrackbar("Y", controls_window, 0, frame_h, nothing)
    cv2.createTrackbar("Width", controls_window, default_w, frame_w, nothing)
    cv2.createTrackbar("Height", controls_window, default_h, frame_h, nothing)

    cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window, 1200, 650)
    cv2.moveWindow(display_window, 0, 0)

    print("Live feed active. Stand in front of the camera and try your gestures.")
    print("Adjust the X, Y, Width, Height sliders in the 'Controls' window to move")
    print("the simulated occlusion over different parts of your body.")
    print("Watch the SKELETON panel to see if detection survives it.")
    print("Press Q to quit.\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera feed lost.")
            break

        x = cv2.getTrackbarPos("X", controls_window)
        y = cv2.getTrackbarPos("Y", controls_window)
        w = cv2.getTrackbarPos("Width", controls_window)
        h = cv2.getTrackbarPos("Height", controls_window)

        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        occluded_frame = frame.copy()
        cv2.rectangle(occluded_frame, (x, y), (x2, y2), OCCLUSION_COLOR, thickness=-1)

        skeleton_frame = draw_skeleton_frame(occluded_frame, pose_model, hands_model)

        left_panel = resize_to_height(occluded_frame, DISPLAY_HEIGHT)
        right_panel = resize_to_height(skeleton_frame, DISPLAY_HEIGHT)
        combined = np.hstack([left_panel, right_panel])

        cv2.putText(combined, "OCCLUDED INPUT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "SKELETON", (left_panel.shape[1] + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(display_window, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()
    hands_model.close()


if __name__ == "__main__":
    main()