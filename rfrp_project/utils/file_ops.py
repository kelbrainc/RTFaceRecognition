#file_ops.py
# utils/file_ops.py

import os, csv
from PIL import Image
import pandas as pd

def save_user_images(name, img1_file, img2_file, dataset_path='dataset'):
    """Save two uploaded images into dataset/<name>/"""
    person_dir = os.path.join(dataset_path, name)
    os.makedirs(person_dir, exist_ok=True)
    for idx, img_file in enumerate([img1_file, img2_file], start=1):
        img = Image.open(img_file)
        img = img.convert('RGB')
        fname = f"{name}_{idx}.jpg"
        img.save(os.path.join(person_dir, fname))

def append_visit(csv_path, timestamp, name, confidence):
    """Append a visit log row."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'name', 'confidence'])
        writer.writerow([timestamp, name, confidence])

def read_visits(csv_path):
    """Load visits.csv as a DataFrame."""
    if not os.path.isfile(csv_path):
        return pd.DataFrame(columns=['timestamp','name','confidence'])
    df = pd.read_csv(csv_path)
    # convert epoch seconds → UTC datetime, then to Singapore time
    df['timestamp'] = (
        pd.to_datetime(df['timestamp'], unit='s', utc=True)
          .dt.tz_convert('Asia/Singapore')
          .dt.tz_localize(None)    # drop tzinfo for simpler display
    )
    return df