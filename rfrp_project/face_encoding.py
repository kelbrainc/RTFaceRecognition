#face_encoding.py

import os
import pickle
import face_recognition
from utils.config import DATASET_PATH, ENCODINGS_PATH

def encode_faces(dataset_path=DATASET_PATH, encodings_path=ENCODINGS_PATH):
    """
    Walks through dataset/<person> folders, computes 128-D embeddings,
    and serializes {'encodings': [...], 'names': [...]} to encodings.pickle.
    """
    known_encodings = []
    known_names = []

    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            # one face per image
            encs = face_recognition.face_encodings(image)
            if len(encs) == 1:
                known_encodings.append(encs[0])
                known_names.append(person)
            else:
                print(f"[WARNING] {img_path}: found {len(encs)} faces; skipping.")

    data = {'encodings': known_encodings, 'names': known_names}
    with open(encodings_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[INFO] Encodings serialized to {encodings_path}.")


if __name__ == '__main__':
    encode_faces()
