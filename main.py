import cv2
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame")
        break

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb)

    # Draw landmarks
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            for landmark in face_landmarks.landmark:

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Face Mesh", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Close if window X is pressed
    if cv2.getWindowProperty("Face Mesh", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()