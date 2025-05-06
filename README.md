# Real-Time Face Recognition System

A clean, Python-based, browser-accessible dashboard for detecting and recognizing faces in a webcam stream—with one-and-done daily logging, live system metrics, and on-the-fly headshot augmentation for people or employees(in business acumen).

## Overview

This project streams your webcam to a Streamlit dashboard. Faces are detected (HOG) and recognized (dlib 128-D embeddings).  
- **One-and-done daily logs**: each known employee is logged the first time they appear today (timestamp + confidence).  
- **Live metrics**: FPS, CPU %, RAM % overlaid on video.  
- **Augment Existing User**: grab N headshots from the live feed for any employee in your dataset—no file dialogs.  
- **Simple UX**: real-time “✅ Access Granted” banner; sidebar for metrics, augmentation, and today’s log.


## Features

- **Live face detection & recognition** (≥10 FPS on CPU)  
- Identifies **known employees** by name (≥ 95 % accuracy on clear, front‐facing images)  
- Marks **unknown visitors** (red box)  
- **Real-time “✅ Access Granted”** status in the **sidebar**   
- **Daily “first‐seen” log** of all recognized employees (timestamp + confidence)  
- **On‐the‐fly headshot augmentation**: grab N fresh images for any existing employee  
- **System metrics overlay** (FPS, CPU %, RAM %) on the video pane  
- **Downloadable CSV** of today’s visit log



## Directory Structure
rfrp_project/
├── dataset/ # per‐employee folders of headshots (for you to add in the images of your employees)
│ └── Billion Man/
│ ├── Billion Man_1.jpg
│ └── Billion Man_2.jpg
├── logs/
│ └── visits.csv # append‐only daily log
├── utils/
│ ├── config.py # thresholds & paths
│ ├── file_ops.py # CSV read/write helpers
│ └── system_metrics.py # CPU/RAM sampling
├── face_detection.py # HOG detector wrapper
├── face_encoding.py # generate encodings.pickle
├── headshot_capture.py # CLI tool to auto‐capture headshots
├── realtime_recognition.py # Streamlit-webrtc VideoProcessor
├── dashboard.py # Streamlit dashboard UI
├── environment.yml # conda spec
├── requirements.txt # pip spec 
└── Makefile # setup / encode / run / dashboard / test / package


## Installation

1. **Clone** the repo  
   ```bash
   git clone https://github.com/kelbrainc/RTFaceRecognition.git
   cd RTFaceRecognition
   cd rfrp_project

2. **Create environment**
    ```bash
    conda env create -f environment.yaml
    conda activate facerecog

or you can
    ```bash
    
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt


3. **Prepare dataset**

- Create subfolders under dataset/ named exactly as each employee (e.g. dataset/Billion Man/).

- Place two clear, front‐facing JPG/PNG images in each folder.


## Usage

1. Generate Face Encodings
Whenever you add or update images in dataset/:
    ```bash
    python face_encoding.py

and it produces encodings.pickle (names + 128-D embeddings).


2. (Optional) Auto-Capture Headshots
Quickly grab N fresh headshots from your webcam for an employee:
    ```bash
    python headshot_capture.py "Billion Man" 2
    python face_encoding.py

this will automatically save 2 new images to dataset/Billion Man/ and re-builds encodings.


3. Launch the Dashboard
    ```bash
    streamlit run dashboard.py

This will open http://localhost:8501 to see:

- Live video with bounding boxes & names (green for known, red for unknown).

- Access Granted banner when a known employee appears.

- System metrics overlaid: FPS, CPU %, RAM %.

Sidebar:

1. Access Status:

“⌛ Starting video…” before camera spins up

“⌛ Waiting for known face…” once streaming but no match

“✅ Name @ HH:MM:SS” as soon as an employee is recognized


2. Augment Existing User:

- Select an employee from the dropdown

- Choose how many headshots to capture (1–5)

- Click Capture Headshots → automatically saves frames to dataset/<Employee>/

A gentle reminder to rerun python face_encoding.py afterwards !

3. Today’s Access Log:

- First-seen timestamp & confidence for each known employee (today only)

- You can download CSV export via button


## Testing
Run unit tests with:
    ```bash
    pytest


## Makefile
1. make setup – create conda env

2. make encode – build encodings.pickle

3. make dashboard – start Streamlit UI

4. make test – run pytest

5. make package – zip project (excludes .git, __pycache__)


## Security & UX Notes
I have made this face recognition system to not allowing new‐user uploads in the dashboard. Enrollment should be made offline or via headshot capture.
Also, unknown visitors are simply shown as “Unknown” (red box) but not logged, and employees are logged only once per day (first time seen).

Feel free to adapt & extend for access control, attendance tracking, or hybrid analytics 😊








