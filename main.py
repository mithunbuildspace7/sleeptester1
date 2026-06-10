import cv2
import mediapipe as mp
import math

# -----------------------------------
# FUNCTIONS
# -----------------------------------

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

def calculate_ear(eye_points):

    vertical1 = distance(eye_points[1], eye_points[5])
    vertical2 = distance(eye_points[2], eye_points[4])

    horizontal = distance(eye_points[0], eye_points[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

# -----------------------------------
# WEBCAM
# -----------------------------------

cap = cv2.VideoCapture(0)

# -----------------------------------
# FACEMESH
# -----------------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Left eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20

# -----------------------------------
# MAIN LOOP
# -----------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        left_eye_points = []
        right_eye_points = []

        # LEFT EYE
        for idx in LEFT_EYE:

            landmark = face.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            left_eye_points.append((x, y))

            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # RIGHT EYE
        for idx in RIGHT_EYE:

            landmark = face.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            right_eye_points.append((x, y))

            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        # Make sure all points exist
        if len(left_eye_points) == 6 and len(right_eye_points) == 6:

            # LEFT EYE LINES
            cv2.line(frame, left_eye_points[1], left_eye_points[5], (0,255,255), 2)
            cv2.line(frame, left_eye_points[2], left_eye_points[4], (0,255,255), 2)
            cv2.line(frame, left_eye_points[0], left_eye_points[3], (0,255,255), 2)

            # RIGHT EYE LINES
            cv2.line(frame, right_eye_points[1], right_eye_points[5], (0,255,255), 2)
            cv2.line(frame, right_eye_points[2], right_eye_points[4], (0,255,255), 2)
            cv2.line(frame, right_eye_points[0], right_eye_points[3], (0,255,255), 2)

            # EAR Calculation
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            ear = (left_ear + right_ear) / 2

            # EAR Display
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Eye State
            if ear < EAR_THRESHOLD:

                cv2.putText(
                    frame,
                    "EYES CLOSED",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            else:

                cv2.putText(
                    frame,
                    "EYES OPEN",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Eye Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("Eye Alert System", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()