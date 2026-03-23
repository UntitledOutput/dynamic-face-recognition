# Dynamic Face Recognition

A real-time facial recognition system built with Python that identifies known individuals from a database and dynamically enrolls new users when an unknown face persists in the camera view for 4 seconds. The system provides instant visual feedback with bounding boxes and personalized greetings for recognized faces.

## Features

- **Real-time Face Detection**: Continuously detects faces in webcam feed using advanced computer vision algorithms.
- **Facial Recognition**: Compares detected faces against a database of known individuals with high accuracy.
- **Dynamic Enrollment**: Automatically prompts for enrollment when an unknown face is detected for 4 seconds, expanding the recognition database on-the-fly.
- **Visual Feedback**: Displays bounding boxes (green for known, red for unknown) and personalized greeting messages.
- **Performance Optimized**: Implements frame resizing and efficient encoding for smooth operation.
- **Robust Handling**: Prevents false enrollments from minor movements and processes one unknown face at a time to maintain stability.

## Tech Stack

- **Python**: Core programming language
- **OpenCV**: Computer vision library for image processing and display
- **face_recognition**: Facial recognition library powered by dlib for accurate face detection and encoding
- **NumPy**: Numerical computing for efficient array operations

## Installation

### Prerequisites

- Python 3.8 or higher installed on your system
- Webcam or camera device connected and accessible

### Steps

1. **Clone or Download the Repository**:
   ```
   git clone https://github.com/emruak86/dynamic-face-recognition.git
   cd dynamic-face-recognition
   ```

2. **Install Dependencies**:
   The project uses a `requirements.txt` file for easy installation. Run:
   ```
   pip install -r requirements.txt
   ```
   This will install OpenCV, face_recognition, NumPy, and other required packages. Note: `face_recognition` requires CMake and may need additional system dependencies on some platforms (e.g., `dlib` compilation).

3. **Prepare Data Directory**:
   Ensure the `data/known_faces/` directory exists. Place initial face images here (one per person, named as `{name}.jpg`).
   ⚠️ Important: The system requires at least one face image in `data/known_faces/` to perform recognition. Without it, all faces will be labeled as "Unknown".

4. **Verify Installation**:
   Run a quick test:
   ```
   python src/webcam_test.py
   ```
   This should open your webcam feed. Press 'q' to exit.

## Usage

### Running the Main Application

Execute the primary script to start the full face recognition system:
```
python src/main.py
```

**What happens:**
- The system loads known faces from `data/known_faces/`.
- Opens webcam and displays live video.
- Detects faces in real-time.
- For known faces: Shows green bounding box, name, and "Hello {name}".
- For unknown faces: Shows red bounding box as "Unknown".
- If an unknown face remains visible for 4 seconds, prompts for a name in the console.
- Saves the face image and adds to the recognition database.
- Press 'q' to exit the application.

### Individual Components (for Development/Testing)

- **Webcam Test** (`src/webcam_test.py`): Basic webcam display to verify camera access.
- **Face Detection** (`src/face_detector.py`): Detects and draws bounding boxes around faces.
- **Face Encoding** (`src/face_encoder.py`): Generates 128D encodings for faces.
- **Load Known Faces** (`src/load_known_face.py`): Loads and encodes faces from the data directory.
- **Enrollment** (`src/enrollment.py`): Dedicated enrollment mode with timer and saving.
- **Recognizer** (`src/recognizer.py`): Core recognition logic (may be integrated into main).

### Adding Known Faces

1. Capture or obtain clear face images (preferably front-facing, well-lit).
2. Save as `{name}.jpg` in `data/known_faces/` (e.g., `john.jpg`).
3. Restart the application to load the new face.

### Troubleshooting

- **Camera not detected**: Ensure webcam is connected and not in use by another application.
- **Import errors**: Verify all dependencies are installed correctly.
- **Low performance**: The system resizes frames for efficiency; ensure adequate hardware.
- **Enrollment not triggering**: Ensure face is clearly visible and still for the 4-second timer.

## Project Structure

```
dynamic-face-recognition/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/
│   └── known_faces/          # Directory for known face images (empty by default, add your own)
│       └── .gitkeep          # Ensures directory is tracked
└── src/
    ├── main.py               # Main application integrating all features
    ├── webcam_test.py        # Basic webcam functionality test
    ├── face_detector.py      # Face detection with bounding boxes
    ├── face_encoder.py       # Face encoding generation
    ├── load_known_face.py    # Load and encode known faces
    ├── enrollment.py         # Dynamic enrollment system
    └── recognizer.py         # Facial recognition logic
```

## Demo

1. **Setup**: Place 1-2 face images in `data/known_faces/` (e.g., your own photo as `yourname.jpg`).
2. **Run**: Execute `python src/main.py`.
3. **Test Recognition**: Show a known face to the camera. Observe green box, name, and greeting.
4. **Test Enrollment**: Have someone unknown appear. Wait 4 seconds for the name prompt. Enter a name. The system will save the face and recognize it immediately afterward.
5. **Exit**: Press 'q' to close the application.

Expected output in console:
```
Loading known faces...
Loaded: yourname
Total known faces: 1
Starting face recognition with enrollment. Unknown faces will be prompted after 4 seconds.
Enrolled: newperson
Program exited.
```

## Future Work

- **User Interface**: Develop a GUI (e.g., using Tkinter or web-based with Flask) for better user interaction instead of console prompts.
- **Database Integration**: Replace file-based storage with a database (SQLite, PostgreSQL) for better management of known faces.
- **Multi-Camera Support**: Extend to handle multiple camera inputs or IP cameras.
- **Mobile Deployment**: Port to mobile platforms using frameworks like Kivy or React Native.
- **Advanced Features**: Add emotion detection, age/gender estimation, or integration with access control systems.
- **System Integration**: Connect Python backend with external systems (e.g., Raspberry Pi rover and C-based display) via API communication.
- **Performance Enhancements**: Implement GPU acceleration or cloud-based processing for higher frame rates.
- **Security**: Add encryption for stored face data and secure enrollment verification.
- **Testing Suite**: Develop comprehensive unit and integration tests for reliability.
