import cv2
import face_recognition
import os
import time
import numpy as np


def handle_enrollment(state, encoding, frame, face_coords, known_encodings, known_names, save_dir):
    """Handle enrollment logic for unknown faces.
    
    Args:
        state: Dictionary with keys 'encoding', 'start', 'last_enrolled'
        encoding: Current face encoding
        frame: Full video frame for saving
        face_coords: Tuple of (top, right, bottom, left) for face location
        known_encodings: List of known encodings (will be modified)
        known_names: List of known names (will be modified)
        save_dir: Directory to save enrolled faces
    
    Returns:
        Tuple of (name, color, state) where name is str, color is tuple, state is dict
    """
    top, right, bottom, left = face_coords
    color = (0, 0, 255)  # Default red for unknown
    name = "Unknown"
    
    if last_enrolled_time := state.get('last_enrolled'):
        if time.time() - last_enrolled_time < 5:
            return name, color, state
    
    if state['encoding'] is None:
        # Start tracking this unknown face
        state['encoding'] = encoding
        state['start'] = time.time()
    else:
        # Compare with tracked encoding
        distance = np.linalg.norm(state['encoding'] - encoding)
        
        if distance < 0.6:
            # Same face → continue timer
            if time.time() - state['start'] >= 4:
                # Prompt for name
                name_input = input("Enter name for this unknown face: ")
                if name_input:
                    # Safe crop
                    h, w, _ = frame.shape
                    top_crop = max(0, top)
                    left_crop = max(0, left)
                    bottom_crop = min(h, bottom)
                    right_crop = min(w, right)

                    face_image = frame[top_crop:bottom_crop, left_crop:right_crop]
                    
                    image_path = os.path.join(save_dir, f"{name_input}.jpg")
                    cv2.imwrite(image_path, face_image)

                    # Reload and encode properly
                    saved_image = face_recognition.load_image_file(image_path)
                    saved_encodings = face_recognition.face_encodings(saved_image)

                    if len(saved_encodings) > 0:
                        known_encodings.append(saved_encodings[0])
                        known_names.append(name_input)
                        print(f"Enrolled: {name_input}")
                        state['last_enrolled'] = time.time()
                    else:
                        print("Failed to encode saved face.")

                # Reset state
                state['encoding'] = None
                state['start'] = None
        else:
            # Different face → reset timer
            state['encoding'] = encoding
            state['start'] = time.time()
    
    return name, color, state


if __name__ == "__main__":
    from load_known_face import load_known_faces
    from face_detector import detect_faces
    from face_encoder import encode_faces
    from recognizer import match_faces

    # Load known faces
    known_encodings, known_names = load_known_faces()
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'known_faces')

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Starting enrollment. Unknown faces will be prompted after 4 seconds.")

    state = {'encoding': None, 'start': None, 'last_enrolled': None}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = detect_faces(rgb_frame)
        face_encodings = encode_faces(rgb_frame, face_locations)

        # Scale back face locations
        face_locations_scaled = [(top*5, right*5, bottom*5, left*5) for (top, right, bottom, left) in face_locations]

        for face_coords, encoding in zip(face_locations_scaled, face_encodings):
            # Try to match
            name = match_faces(encoding, known_encodings, known_names)
            
            if name != "Unknown":
                # Known face
                color = (0, 255, 0)
                state['encoding'] = None  # Reset timer
            else:
                # Unknown face - handle enrollment
                name, color, state = handle_enrollment(state, encoding, frame, face_coords, 
                                                        known_encodings, known_names, save_dir)

            # Draw box and label
            top, right, bottom, left = face_coords
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Enrollment exited.")
