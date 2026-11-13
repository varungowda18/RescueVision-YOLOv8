"""
Mode 2: Security Monitoring (DeepFace version - no dlib/compiler needed)

Loads a reference image (e.g. your friend's photo), then scans a live
camera feed - either your laptop webcam OR a phone acting as an IP camera.
Faces are detected with OpenCV's Haar cascade, cropped, and compared
against the reference photo using DeepFace.verify(). On a match, flags it
on screen and plays alarm.wav (shared from the project root).

Install (only for this mode):
    pip install -r requirements_mode2.txt

Using your phone as an IP camera:
    1. Install "IP Webcam" (Android) or "DroidCam" / similar (iOS).
    2. Connect your phone to the SAME Wi-Fi network as this computer.
    3. Open the app and start the server - it will show a URL like:
           http://192.168.1.5:8080/video
    4. Pick "IP Camera" when this script asks, and paste that URL.

First run will download DeepFace's model weights automatically (~100MB,
needs internet once).
"""

import cv2
import numpy as np
import os
import time
import threading
from deepface import DeepFace

try:
    from playsound import playsound
except ImportError:
    playsound = None

# --- Paths (relative to this file) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # victimproject/
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_faces")
ALARM_PATH = os.path.join(PROJECT_ROOT, "alarm.wav")  # reuse Mode 1's alarm.wav

MODEL_NAME = "Facenet512"     # stronger model, more accurate than Facenet
DISTANCE_METRIC = "cosine"
COOLDOWN_SECONDS = 5           # avoid re-triggering alarm every frame
PROCESS_EVERY_N = 5            # skip frames for speed (DeepFace is heavier than a plain crop)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------------------------------------------------------------------
# Threaded camera reader: always keeps the LATEST frame only, discarding
# stale ones. This is what fixes the glitching/lag - without it, frames pile
# up in a buffer while DeepFace is busy, and you end up watching the past.
# Especially important for IP camera streams which buffer more aggressively.
# ---------------------------------------------------------------------------
class CameraStream:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")

        # keep OpenCV's own buffer as small as possible (helps on sources that support it)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def play_alarm():
    if playsound and os.path.exists(ALARM_PATH):
        threading.Thread(target=playsound, args=(ALARM_PATH,), daemon=True).start()


def list_reference_images():
    if not os.path.exists(REFERENCE_DIR):
        return []
    valid_ext = (".jpg", ".jpeg", ".png")
    return [f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith(valid_ext)]


def choose_reference_image():
    images = list_reference_images()
    if not images:
        raise FileNotFoundError(
            f"No images found in {REFERENCE_DIR}. Add a photo there first."
        )
    if len(images) == 1:
        print(f"[INFO] Using reference image: {images[0]}")
        return os.path.join(REFERENCE_DIR, images[0])

    print("Available reference images:")
    for i, name in enumerate(images, start=1):
        print(f"  {i}. {name}")
    idx = int(input("Choose reference image number: ").strip()) - 1
    return os.path.join(REFERENCE_DIR, images[idx])


def choose_camera_source():
    print("\nSelect camera source:")
    print("  1. Laptop webcam")
    print("  2. IP camera (phone)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        url = input(
            "Enter the IP camera URL (e.g. http://192.168.1.5:8080/video): "
        ).strip()
        return url
    else:
        return 0  # default webcam index


def verify_face(face_crop_path, ref_path):
    """Returns (is_match, distance) using DeepFace.verify, or (False, None) on failure."""
    try:
        result = DeepFace.verify(
            img1_path=ref_path,
            img2_path=face_crop_path,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
        )
        return result["verified"], result["distance"]
    except Exception as e:
        print(f"[DEBUG] verify failed: {e}")
        return False, None


def verification_worker(state, ref_path, temp_crop_path):
    """
    Runs in its own thread. Waits for a new face crop to appear, verifies it
    against the reference, and stores the result. Never touches the display
    loop directly, so the video keeps playing smoothly no matter how long
    DeepFace takes.
    """
    while state["running"]:
        with state["lock"]:
            job = state["pending_job"]
            state["pending_job"] = None

        if job is None:
            time.sleep(0.02)
            continue

        (x, y, w, h, face_crop) = job
        cv2.imwrite(temp_crop_path, face_crop)
        is_match, distance = verify_face(temp_crop_path, ref_path)

        if distance is not None:
            print(f"[DEBUG] distance={distance:.3f}  match={is_match}")

        with state["lock"]:
            state["last_results"] = [(x, y, w, h, is_match, distance)]

        if is_match:
            now = time.time()
            if now - state["last_alarm_time"] > COOLDOWN_SECONDS:
                print("[ALERT] Target person detected!")
                play_alarm()
                state["last_alarm_time"] = now


def run_security_monitor():
    ref_path = choose_reference_image()
    source = choose_camera_source()

    print("[INFO] Reference image ready.")
    print(f"[INFO] Connecting to camera source: {source}")

    stream = CameraStream(source)

    frame_count = 0
    temp_crop_path = os.path.join(BASE_DIR, "_temp_face_crop.jpg")

    # shared state between main display loop and the background verification thread
    state = {
        "running": True,
        "lock": threading.Lock(),
        "pending_job": None,     # (x, y, w, h, face_crop) waiting to be verified
        "last_results": [],      # most recent (x, y, w, h, is_match, distance) to draw
        "last_alarm_time": 0,
    }

    worker = threading.Thread(
        target=verification_worker, args=(state, ref_path, temp_crop_path), daemon=True
    )
    worker.start()

    print("[INFO] Starting security monitoring. Press 'q' to quit.")

    while True:
        ret, frame = stream.read()
        if not ret:
            # camera not ready yet (common right after connecting to an IP stream)
            time.sleep(0.05)
            continue

        frame_count += 1
        display_frame = frame.copy()

        # Face detection (Haar cascade) is cheap - fine to run on the main thread.
        if frame_count % PROCESS_EVERY_N == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces) > 0:
                x, y, w, h = faces[0]  # track the largest/first face for now
                pad = int(0.2 * w)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    # hand off the heavy DeepFace work to the background thread;
                    # if it's still busy with a previous job, just skip this one
                    with state["lock"]:
                        if state["pending_job"] is None:
                            state["pending_job"] = (x, y, w, h, face_crop)

        # draw whatever the worker thread last figured out - never blocks
        with state["lock"]:
            results = list(state["last_results"])

        for (x, y, w, h, is_match, distance) in results:
            color = (0, 0, 255) if is_match else (0, 255, 0)
            dist_str = f"{distance:.2f}" if distance is not None else "?"
            label = f"MATCH ({dist_str})" if is_match else f"Unknown ({dist_str})"
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(display_frame, "Mode 2: Security Monitoring", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Security Monitoring", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    state["running"] = False
    stream.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_crop_path):
        os.remove(temp_crop_path)


if __name__ == "__main__":
    run_security_monitor()