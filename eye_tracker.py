"""
Gaze zone + blink detection from a plain webcam.

v1 scope (deliberately small):
  - detect the face
  - classify gaze into 5 zones: CENTER, LEFT, RIGHT, UP, DOWN
  - detect blinks

Explicitly NOT in v1: precise cursor control. Gaze estimation from a
consumer webcam is not accurate enough for pixel coordinates, so this
uses coarse zones instead. That is a design decision, not a limitation
to fix later.

Run:  python eye_tracker.py        (q to quit, c to recalibrate center)
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe removed the old `mp.solutions` API. The current Tasks API needs a
# model file, downloaded once on first run and reused after that.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

# --- MediaPipe Face Mesh landmark indices ---------------------------------
# The Face Landmarker returns 478 points; 468-477 are the iris.
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corners, used to normalise iris position against eye width/height.
LEFT_EYE_CORNERS = (33, 133)      # outer, inner
RIGHT_EYE_CORNERS = (362, 263)    # inner, outer

# Points used for the eye aspect ratio (blink detection).
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# --- Tuning knobs ---------------------------------------------------------
EAR_THRESHOLD = 0.21          # below this, the eye counts as closed
BLINK_MIN_FRAMES = 2          # frames closed before it counts as a blink
GAZE_MARGIN_X = 0.09          # how far from centre before it reads as L/R
GAZE_MARGIN_Y = 0.06          # same, vertically
SMOOTHING = 5                 # frames averaged to reduce jitter


def landmarks_to_array(landmarks, width, height):
    """Convert normalised MediaPipe landmarks to pixel coordinates."""
    return np.array(
        [(lm.x * width, lm.y * height) for lm in landmarks],
        dtype=np.float32,
    )


def eye_aspect_ratio(points, indices):
    """
    Vertical eye opening divided by horizontal eye width.

    Drops when the eyelid closes, which makes it a cheap and reliable
    blink signal that does not depend on lighting or skin tone.
    """
    p0, p1, p2, p3, p4, p5 = (points[i] for i in indices)
    vertical = np.linalg.norm(p1 - p5) + np.linalg.norm(p2 - p4)
    horizontal = 2.0 * np.linalg.norm(p0 - p3)
    return vertical / horizontal if horizontal else 0.0


def iris_offset(points, iris_indices, corner_indices):
    """
    Where the iris sits inside the eye, as a fraction from -0.5 to 0.5.

    0 means centred. Normalising against the eye's own width is what makes
    this survive the user moving closer to or further from the camera.
    """
    iris = points[iris_indices].mean(axis=0)
    outer, inner = points[corner_indices[0]], points[corner_indices[1]]

    eye_centre = (outer + inner) / 2.0
    eye_width = np.linalg.norm(inner - outer)
    if eye_width == 0:
        return 0.0, 0.0

    dx = (iris[0] - eye_centre[0]) / eye_width
    dy = (iris[1] - eye_centre[1]) / eye_width
    return dx, dy


def classify_gaze(dx, dy, calib_x, calib_y):
    """Map a smoothed iris offset to one of five zones."""
    dx -= calib_x
    dy -= calib_y

    if dy < -GAZE_MARGIN_Y:
        return "UP"
    if dy > GAZE_MARGIN_Y:
        return "DOWN"
    if dx < -GAZE_MARGIN_X:
        return "LEFT"
    if dx > GAZE_MARGIN_X:
        return "RIGHT"
    return "CENTER"


def ensure_model():
    """Download the face landmark model the first time the script runs."""
    if MODEL_PATH.exists():
        return

    print("Downloading face landmark model (one time, a few MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        # Leave no half-written file behind, or the next run will load a
        # truncated model and fail somewhere much less obvious.
        MODEL_PATH.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not download the face landmark model: {exc}\n"
            f"Check your internet connection, or download it manually from\n"
            f"  {MODEL_URL}\n"
            f"and save it as {MODEL_PATH.name} next to this script."
        )
    print("Model saved to", MODEL_PATH)


def create_landmarker():
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,   # synchronous, one frame at a time
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No camera found. Check that nothing else is using it.")

    landmarker = create_landmarker()
    last_ts = -1   # VIDEO mode requires strictly increasing timestamps

    history: list[tuple[float, float]] = []
    calib_x = calib_y = 0.0
    sx = sy = 0.0          # latest smoothed offset, read by the calibrate key
    closed_frames = 0
    blink_count = 0
    last_blink_at = 0.0
    gaze = "NO FACE"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          # mirror, so it reads like a mirror
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int(time.monotonic() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        result = landmarker.detect_for_video(mp_image, ts)

        if result.face_landmarks:
            points = landmarks_to_array(result.face_landmarks[0], width, height)

            # --- blink ---
            ear = (
                eye_aspect_ratio(points, LEFT_EYE_EAR)
                + eye_aspect_ratio(points, RIGHT_EYE_EAR)
            ) / 2.0

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                if closed_frames >= BLINK_MIN_FRAMES:
                    blink_count += 1
                    last_blink_at = time.time()
                closed_frames = 0

            # --- gaze ---
            lx, ly = iris_offset(points, LEFT_IRIS, LEFT_EYE_CORNERS)
            rx, ry = iris_offset(points, RIGHT_IRIS, RIGHT_EYE_CORNERS)

            history.append(((lx + rx) / 2.0, (ly + ry) / 2.0))
            history = history[-SMOOTHING:]
            sx = sum(p[0] for p in history) / len(history)
            sy = sum(p[1] for p in history) / len(history)

            gaze = "BLINK" if ear < EAR_THRESHOLD else classify_gaze(
                sx, sy, calib_x, calib_y
            )

            for idx in LEFT_IRIS + RIGHT_IRIS:
                cv2.circle(frame, tuple(points[idx].astype(int)), 2, (0, 255, 0), -1)

            cv2.putText(
                frame, f"EAR {ear:.3f}", (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
            )
        else:
            gaze = "NO FACE"
            history.clear()

        recent_blink = time.time() - last_blink_at < 0.4
        colour = (0, 200, 255) if recent_blink else (255, 255, 255)
        cv2.putText(frame, gaze, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2)
        cv2.putText(
            frame, f"blinks: {blink_count}   [c] calibrate   [q] quit",
            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
        )

        cv2.imshow("eye-tracker v1", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and history:
            calib_x, calib_y = sx, sy   # treat current gaze as dead centre

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
