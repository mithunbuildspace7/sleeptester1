import cv2
import mediapipe as mp
import math
import time
import pygame

# -----------------------------
# PYGAME ALARM SETUP
# -----------------------------

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# -----------------------------
# FUNCTIONS
# -----------------------------

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

def calculate_ear(eye_points):

    vertical1 = distance(eye_points[1], eye_points[5])
    vertical2 = distance(eye_points[2], eye_points[4])

    horizontal = distance(eye_points[0], eye_points[3])

    return (vertical1 + vertical2) / (2.0 * horizontal)

# -----------------------------
# FACEMESH
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -----------------------------
# EYE LANDMARKS
# -----------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -----------------------------
# SETTINGS
# -----------------------------

EAR_THRESHOLD = 0.20

closed_start_time = None
alarm_playing = False

# -----------------------------
# WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)

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

        if len(left_eye_points) == 6 and len(right_eye_points) == 6:

            # EAR LINES

            cv2.line(frame, left_eye_points[1], left_eye_points[5], (0,255,255), 2)
            cv2.line(frame, left_eye_points[2], left_eye_points[4], (0,255,255), 2)
            cv2.line(frame, left_eye_points[0], left_eye_points[3], (0,255,255), 2)

            cv2.line(frame, right_eye_points[1], right_eye_points[5], (0,255,255), 2)
            cv2.line(frame, right_eye_points[2], right_eye_points[4], (0,255,255), 2)
            cv2.line(frame, right_eye_points[0], right_eye_points[3], (0,255,255), 2)

            # EAR

            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            ear = (left_ear + right_ear) / 2

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # EYES CLOSED

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

                if closed_start_time is None:
                    closed_start_time = time.time()

                elapsed = time.time() - closed_start_time

                cv2.putText(
                    frame,
                    f"Timer: {elapsed:.1f}s",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                # START ALARM AFTER 3 SECONDS

                if elapsed >= 2:

                    if not alarm_playing:

                        pygame.mixer.music.play(-1)

                        alarm_playing = True

            # EYES OPEN

            else:

                closed_start_time = None

                cv2.putText(
                    frame,
                    "EYES OPEN",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                if alarm_playing:

                    pygame.mixer.music.stop()

                    alarm_playing = False

    cv2.imshow("Eye Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("Eye Alert System", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()

pygame.mixer.music.stop()
pygame.quit()

cv2.destroyAllWindows()