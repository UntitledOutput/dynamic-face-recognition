import numpy as np
import face_recognition
import cv2
from load_known_face import load_known_faces
from face_detector import detect_faces
from face_encoder import encode_faces


def match_faces(encoding, known_encodings, known_names):
    """Match a face encoding against known encodings.
    
    Args:
        encoding: Face encoding to match
        known_encodings: List of known face encodings
        known_names: List of names corresponding to known encodings
    
    Returns:
        String: Name of matched face or "Unknown"
    """
    if not known_encodings:
        return "Unknown"
    
    face_distances = face_recognition.face_distance(known_encodings, encoding)
    best_match_index = np.argmin(face_distances)
    
    if face_distances[best_match_index] < 0.6:
        return known_names[best_match_index]
    else:
        return "Unknown"


if __name__ == "__main__":
    # Load known faces
    known_encodings, known_names = load_known_faces()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Webcam opened successfully. Starting face recognition...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = detect_faces(rgb_frame)

        # Encode faces
        face_encodings = encode_faces(rgb_frame, face_locations)

        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5

            # Match against known encodings
            name = match_faces(face_encoding, known_encodings, known_names)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Display result
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")
