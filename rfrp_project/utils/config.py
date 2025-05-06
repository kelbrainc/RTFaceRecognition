#config.py
# utils/config.py

# Detection model: 'haar' or 'hog'
DETECTION_MODEL = 'hog'

# Face-recognition threshold
RECOGNITION_THRESHOLD = 0.55

# Video frame size
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Paths
DATASET_PATH = 'dataset'
ENCODINGS_PATH = 'encodings.pickle'
LOG_PATH = 'logs/visits.csv'
