import cv2
import os
import time
from load_known_face import load_known_faces
from face_detector import detect_faces
from face_encoder import encode_faces
from recognizer import match_faces
from enrollment import handle_enrollment


def main():
    """Main face recognition with enrollment pipeline."""
    # Load known faces
    known_encodings, known_names = load_known_faces()
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'known_faces')

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting face recognition with enrollment. Unknown faces will be prompted after 4 seconds.")

    # State for tracking unknown faces
    state = {
        'encoding': None,
        'start': None,
        'last_enrolled': None
    }
    handled_unknown = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = detect_faces(rgb_frame)
        
        # Encode faces
        face_encodings = encode_faces(rgb_frame, face_locations)

        # Scale back face locations
        face_locations_scaled = [(top*5, right*5, bottom*5, left*5) for (top, right, bottom, left) in face_locations]

        handled_unknown = False

        for face_coords, encoding in zip(face_locations_scaled, face_encodings):
            top, right, bottom, left = face_coords
            name = "Unknown"
            greeting = ""
            color = (0, 0, 255)  # Default red

            # Try to match face
            matched_name = match_faces(encoding, known_encodings, known_names)
            
            if matched_name != "Unknown":
                # Known face
                name = matched_name
                color = (0, 255, 0)  # Green
                state['encoding'] = None  # Reset enrollment timer
                state['start'] = None
                greeting = f"Hello {name}"
            else:
                # Unknown face - handle enrollment
                if not handled_unknown:
                    handled_unknown = True
                    name, color, state = handle_enrollment(state, encoding, frame, face_coords,
                                                            known_encodings, known_names, save_dir)

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw name
            cv2.putText(frame, name, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw greeting if known
            if greeting:
                cv2.putText(frame, greeting, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program exited.")


if __name__ == "__main__":
    main()
