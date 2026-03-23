import cv2
import face_recognition


def encode_faces(rgb_frame, face_locations):
    """Encode faces in an RGB frame.
    
    Args:
        rgb_frame: Frame in RGB format (not BGR)
        face_locations: List of face locations from detect_faces()
    
    Returns:
        List of face encodings (128-dimensional vectors)
    """
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_encodings


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Webcam opened successfully. Starting face encoding...")

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
        from face_detector import detect_faces
        face_locations = detect_faces(rgb_frame)
        
        print("Faces detected:", len(face_locations))

        # Generate encodings
        face_encodings = encode_faces(rgb_frame, face_locations)
        
        # Scale back face locations for display
        face_locations_scaled = [(top*5, right*5, bottom*5, left*5) for (top, right, bottom, left) in face_locations]
        
        # Draw bounding boxes
        for top, right, bottom, left in face_locations_scaled:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Print encoding info
        if face_encodings:
            print(f"Face encoding generated: {len(face_encodings[0])} dimensions")

        cv2.imshow("Face Encoder", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")
