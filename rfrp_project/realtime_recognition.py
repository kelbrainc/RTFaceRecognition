#realtime_recognition.py

import time
import av
import cv2
import pickle
import numpy as np
import face_recognition
from streamlit_webrtc import VideoProcessorBase

from face_detection import FaceDetector
from utils.config import (
    ENCODINGS_PATH,
    LOG_PATH,
    RECOGNITION_THRESHOLD,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
)
from utils.file_ops import append_visit
from utils.system_metrics import get_system_metrics


class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        # Load detector & encodings once
        self.detector = FaceDetector()
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        self.known_encodings = data["encodings"]
        self.known_names     = data["names"]

        # FPS tracking
        self._prev_time = time.time()
        self.fps = 0.0

        # Exposed to dashboard
        self.last_name      = None
        self.last_conf      = 0.0
        self.last_face_crop = None

        # Track who’s already been logged
        self._logged = set()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1) Grab & resize
        img = frame.to_ndarray(format="bgr24")
        small = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))

        # 2) Detect + encode
        locs = self.detector.detect(small)
        encs = face_recognition.face_encodings(small, locs)

        # 3) Identify first face only for logging/banner
        name, conf = None, 0.0
        if encs:
            dists = face_recognition.face_distance(self.known_encodings, encs[0])
            if len(dists) and np.min(dists) < RECOGNITION_THRESHOLD:
                idx = int(np.argmin(dists))
                name = self.known_names[idx]
                conf = 1.0 - float(dists[idx])

        self.last_name = name
        self.last_conf = conf

        # 4) Log once per known employee
        if name and name not in self._logged:
            append_visit(LOG_PATH, time.time(), name, conf)
            self._logged.add(name)

        # 5) Draw bounding boxes & labels; save last crop
        for ((top, right, bottom, left), face_enc) in zip(locs, encs):
            d = face_recognition.face_distance(self.known_encodings, face_enc)
            if len(d) and np.min(d) < RECOGNITION_THRESHOLD:
                i = int(np.argmin(d))
                fn = self.known_names[i]
                fc = 1.0 - float(d[i])
                color = (0,255,0)
                label = f"{fn} ({fc:.2f})"
            else:
                color = (0,0,255)
                label = "Unknown"

            cv2.rectangle(small, (left, top), (right, bottom), color, 2)
            cv2.putText(
                small, label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            self.last_face_crop = small[top:bottom, left:right].copy()

        # 6) FPS & metrics
        now = time.time()
        delta = now - self._prev_time
        self.fps = 1.0 / delta if delta > 0 else 0.0
        self._prev_time = now
        cpu, mem = get_system_metrics()

        # 7) Overlay metrics on-frame
        cv2.putText(small, f"FPS: {self.fps:.1f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(small, f"CPU: {cpu:.0f}%", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(small, f"RAM: {mem:.0f}%", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return av.VideoFrame.from_ndarray(small, format="bgr24")
