#face_detection.py

import face_recognition
from utils.config import DETECTION_MODEL

class FaceDetector:
    def __init__(self, model=DETECTION_MODEL):
        self.model = 'hog'

    def detect(self, frame):
        """
        Input: BGR frame (numpy array).
        Returns: list of (top, right, bottom, left) boxes.
        """
        # face_recognition needs RGB
        rgb = frame[:, :, ::-1]
        # HOG-based face_locations
        return face_recognition.face_locations(rgb, model='hog')



