#dashboard.py

import os
import cv2
import pickle
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from realtime_recognition import FaceRecognitionProcessor
from utils.config import DATASET_PATH, LOG_PATH
from utils.file_ops import read_visits, save_user_images
from utils.system_metrics import get_system_metrics

# --- Page & Streamlit setup ---
st.set_page_config(page_title="Face Recognition Dashboard", layout="wide")
st.title("Real-Time Face Recognition")

# --- Start WebRTC stream ---
RTC_CONF = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
webrtc_ctx = webrtc_streamer(
    key="face-recog",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONF,
    video_processor_factory=FaceRecognitionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
proc = webrtc_ctx.video_processor  # VideoProcessor instance


# 1) Sidebar: Access Status
st.sidebar.header("Access Status")
if webrtc_ctx.state.playing and proc:
    name = getattr(proc, "last_name", None)
    if name:
        ts = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.success(f"✅ {name} @ {ts}")
    else:
        st.sidebar.info("⌛ Waiting for known face…")
else:
    st.sidebar.info("⌛ Starting video…")


# 2) Sidebar: Augment Existing User (capture headshots)
st.sidebar.header("Augment Existing User")
employees = [
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]
emp = st.sidebar.selectbox("Employee", [""] + employees)
count = st.sidebar.slider("How many headshots?", 1, 5, 2)
if st.sidebar.button("Capture Headshots"):
    if not emp:
        st.sidebar.warning("Please select an employee.")
    elif not proc or proc.last_face_crop is None:
        st.sidebar.warning("No face detected yet.")
    else:
        saved = 0
        start = time.time()
        st.sidebar.info(f"Capturing {count} for '{emp}'…")
        while saved < count and (time.time() - start) < 10:
            crop = proc.last_face_crop
            if crop is not None:
                ts = int(time.time())
                fn = f"{emp}_{ts}.jpg"
                folder = os.path.join(DATASET_PATH, emp)
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, fn), crop)
                st.sidebar.success(f"✔ {fn}")
                saved += 1
                time.sleep(0.5)
        if saved < count:
            st.sidebar.error(f"Only captured {saved}/{count}.")
        else:
            st.sidebar.success(f"Captured {count} headshots.")
            st.sidebar.info("→ Run `python face_encoding.py` to re-encode.")


# 3) Sidebar: Today's Access Log (unique first-seen)
st.sidebar.header("Today's Access Log")
if os.path.exists(LOG_PATH):
    df = read_visits(LOG_PATH)
    today = datetime.now(ZoneInfo("Asia/Singapore")).date()
    df = df[df["timestamp"].dt.date == today]
else:
    df = pd.DataFrame(columns=["timestamp", "name", "confidence"])

if not df.empty:
    display = df.copy()
    display["timestamp"] = display["timestamp"].dt.strftime("%H:%M:%S")
    # only first-seen per person
    firsts = display.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    st.sidebar.table(firsts[["timestamp", "name", "confidence"]])
    csv = df.to_csv(index=False).encode()
    st.sidebar.download_button("Download CSV", csv, "visits_today.csv", "text/csv")
else:
    st.sidebar.write("No entries logged today.")

