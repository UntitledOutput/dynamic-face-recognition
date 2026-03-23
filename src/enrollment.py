import cv2
import face_recognition
import os
import time
import numpy as np

# Load known faces
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

print(f"Total known faces: {len(known_names)}")

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting enrollment. Unknown faces will be prompted after 4 seconds.")

unknown_start_time = None
unknown_encoding = None
last_enrolled_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Scale back face locations
    face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]

    handled_unknown = False

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            # Known face
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            color = (0, 255, 0)  # Green
            unknown_start_time = None  # Reset timer
            unknown_encoding = None
        else:
            # Unknown face
            color = (0, 0, 255)  # Red
            if not handled_unknown:
                handled_unknown = True
                if last_enrolled_time and time.time() - last_enrolled_time < 5:
                    continue
                if unknown_encoding is None:
                    unknown_encoding = encoding
                    unknown_start_time = time.time()
                else:
                    distance = np.linalg.norm(unknown_encoding - encoding)
                    if distance < 0.6:
                        # Same face → continue timer
                        if time.time() - unknown_start_time >= 4:
                            # Prompt for name
                            name_input = input("Enter name for this unknown face: ")
                            if name_input:
                                # Save only the face
                                face_image = frame[top:bottom, left:right]
                                image_path = os.path.join(KNOWN_FACES_DIR, f"{name_input}.jpg")
                                cv2.imwrite(image_path, face_image)
                                print(f"Enrolled: {name_input}")
                                # Append to known
                                known_encodings.append(unknown_encoding)
                                known_names.append(name_input)
                                last_enrolled_time = time.time()
                            unknown_start_time = None
                            unknown_encoding = None
                    else:
                        # Different face → reset timer
                        unknown_encoding = encoding
                        unknown_start_time = time.time()

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Enrollment", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Enrollment exited.")
