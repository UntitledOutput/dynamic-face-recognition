import face_recognition
import os

KNOWN_FACES_DIR = r'C:\Users\emran\desktop\dynamic-face-recognition\data\known_faces'

known_encodings = []
known_names = []

print("Loading known faces...")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            name = os.path.splitext(filename)[0]
            known_encodings.append(encoding)
            known_names.append(name)
            print(f"Loaded: {name}")

print(f"Total faces loaded: {len(known_names)}")
