import face_recognition
import os


def load_known_faces(directory=None):
    """Load known faces from a directory.
    
    Args:
        directory: Path to known faces directory. If None, uses relative path.
    
    Returns:
        Tuple of (known_encodings, known_names)
    """
    if directory is None:
        directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'known_faces')
    
    known_encodings = []
    known_names = []

    if not os.path.exists(directory):
        print(f"Known faces directory not found: {directory}")
        return known_encodings, known_names

    print("Loading known faces...")

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                encoding = encodings[0]
                name = os.path.splitext(filename)[0]
                known_encodings.append(encoding)
                known_names.append(name)
                print(f"Loaded: {name}")

    print(f"Total faces loaded: {len(known_names)}")
    return known_encodings, known_names


if __name__ == "__main__":
    known_encodings, known_names = load_known_faces()
    print(f"\nLoaded {len(known_names)} faces total.")
