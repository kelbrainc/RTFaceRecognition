#headshot_cpature.py

import os
import sys
import time
import cv2
import face_recognition
from utils.config import DATASET_PATH

def auto_capture(name: str, count: int = 2, cam_index: int = 0, timeout: float = 10.0):
    """
    Captures `count` face crops from the webcam and saves them under dataset/<name>/.
    Uses DirectShow backend on Windows and times out after `timeout` seconds if no frames arrive.
    """
    out_dir = os.path.join(DATASET_PATH, name)
    os.makedirs(out_dir, exist_ok=True)

    # Use DirectShow on Windows to avoid MSMF grabFrame errors
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_index}.")
        return

    saved = 0
    start = time.time()

    print(f"[INFO] Capturing {count} headshots for '{name}'...")

    while saved < count and time.time() - start < timeout:
        ret, frame = cap.read()
        if not ret:
            # no frame yet; wait a bit and retry
            time.sleep(0.1)
            continue

        # detect the first face in the frame
        locs = face_recognition.face_locations(frame, model='hog')
        if locs:
            top, right, bottom, left = locs[0]
            crop = frame[top:bottom, left:right]
            fname = f"{name}_{saved+1}.jpg"
            path  = os.path.join(out_dir, fname)
            cv2.imwrite(path, crop)
            print(f"[INFO] Saved {fname}")
            saved += 1
            time.sleep(0.5)  # avoid duplicates

        # optional: show live feedback
        cv2.imshow("Auto-Capture Headshots (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved < count:
        print(f"[WARN] Only captured {saved}/{count} images before timeout.")
    else:
        print("[INFO] Done capturing headshots.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python headshot_capture.py \"Full Name\" [count]")
        sys.exit(1)
    nm = sys.argv[1]
    cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    auto_capture(nm, cnt)
