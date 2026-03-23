import cv2
import face_recognition


def detect_faces(rgb_frame):
    """Detect faces in an RGB frame.
    
    Args:
        rgb_frame: Frame in RGB format (not BGR)
    
    Returns:
        List of face locations (top, right, bottom, left)
    """
    face_locations = face_recognition.face_locations(rgb_frame)
    return face_locations


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Webcam opened successfully. Starting face detection...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = detect_faces(rgb_frame)

        # Draw bounding boxes
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")
