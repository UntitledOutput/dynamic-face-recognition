import cv2
import face_recognition

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

    # Resize frame for faster processing (optional) 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    
    print("Faces detected:", len(face_locations))

    # Generate encodings
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Draw bounding boxes
    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

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
