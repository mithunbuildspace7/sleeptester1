import cv2
import mediapipe as mp
import math
import time
import pygame
import numpy as np

# -----------------------------
# PYGAME ALARM SETUP
# -----------------------------

pygame.mixer.init()
try:
    pygame.mixer.music.load("alarm.wav")
except:
    pass

# -----------------------------
# DESIGN TOKENS
# -----------------------------

# Color palette (BGR format for OpenCV)
COLOR_BG_DARK       = (15, 15, 20)          # near-black panel bg
COLOR_ACCENT_CYAN   = (220, 200, 50)        # cyan-ish (BGR)
COLOR_ACCENT_AMBER  = (0, 165, 245)         # amber warning (BGR)
COLOR_ACCENT_RED    = (50, 50, 230)         # danger red (BGR)
COLOR_ACCENT_GREEN  = (100, 210, 80)        # open/safe green (BGR)
COLOR_WHITE         = (240, 240, 240)
COLOR_MUTED         = (120, 118, 110)
COLOR_PANEL_BG      = (28, 26, 22)          # dark warm panel

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_DUPLEX

# -----------------------------
# FUNCTIONS
# -----------------------------

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_ear(eye_points):
    vertical1  = distance(eye_points[1], eye_points[5])
    vertical2  = distance(eye_points[2], eye_points[4])
    horizontal = distance(eye_points[0], eye_points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)
def calculate_mar(top, bottom, left, right):

    vertical = distance(top, bottom)
    horizontal = distance(left, right)

    return vertical / horizontal

def draw_rounded_rect(frame, x, y, w, h, radius, color, thickness=-1, alpha=1.0):
    """Draw a rounded rectangle, optionally with alpha blending."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, thickness)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, thickness)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, thickness)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_panel(frame, x, y, w, h, alpha=0.75):
    """Semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Thin border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 48, 44), 1)

def draw_eye_landmark(frame, points, color, line_color, radius=4):
    """Draw eye landmarks with connecting lines."""
    # Draw lines connecting landmarks
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        cv2.line(frame, p1, p2, (*line_color, 80), 1, cv2.LINE_AA)
    # Draw landmark dots
    for pt in points:
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

def draw_ear_lines(frame, pts, color):
    """Draw EAR measurement lines with clean styling."""
    # Vertical lines (eye openness)
    cv2.line(frame, pts[1], pts[5], color, 1, cv2.LINE_AA)
    cv2.line(frame, pts[2], pts[4], color, 1, cv2.LINE_AA)
    # Horizontal line (eye width)
    cv2.line(frame, pts[0], pts[3], COLOR_MUTED, 1, cv2.LINE_AA)

def draw_arc_progress(frame, cx, cy, radius, progress, color, thickness=6):
    """Draw a circular arc progress indicator."""
    start_angle = -90
    end_angle   = start_angle + int(360 * progress)
    # Background arc
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                0, 360, (50, 48, 44), thickness, cv2.LINE_AA)
    # Foreground arc
    if progress > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                    start_angle, end_angle, color, thickness, cv2.LINE_AA)

def lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def put_text_centered(frame, text, cx, y, font, scale, color, thickness=1):
    """Draw text centered at cx."""
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, (cx - tw // 2, y), font, scale, color, thickness, cv2.LINE_AA)

def put_label(frame, text, x, y, color, scale=0.45, thickness=1):
    """Uppercase small label."""
    cv2.putText(frame, text.upper(), (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

# -----------------------------
# FACEMESH SETUP
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP  = 13
LOWER_LIP  = 14

LEFT_MOUTH  = 78
RIGHT_MOUTH = 308

# -----------------------------
# SETTINGS
# -----------------------------

EAR_THRESHOLD   = 0.2
ALARM_DELAY_SEC = 1.5
MAR_THRESHOLD = 0.55
mar = 0.0
closed_start_time = None
alarm_playing     = False

# Smoothed EAR for stable display
ear_smooth = 0.25

# -----------------------------
# WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror
    h, w, _ = frame.shape

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Default state
    eyes_open     = True
    left_ear_val  = 0.0
    right_ear_val = 0.0
    ear           = 0.0
    elapsed       = 0.0
    progress      = 0.0
    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True
        face = results.multi_face_landmarks[0]

        #face landmarks
        upper_lip  = face.landmark[UPPER_LIP]
        lower_lip  = face.landmark[LOWER_LIP]

        left_mouth  = face.landmark[LEFT_MOUTH]
        right_mouth = face.landmark[RIGHT_MOUTH]
        #conversion of pixels
        top_mouth = (
            int(upper_lip.x * w),
            int(upper_lip.y * h)
        )

        bottom_mouth = (
            int(lower_lip.x * w),
            int(lower_lip.y * h)
        )

        left_mouth_pt = (
            int(left_mouth.x * w),
            int(left_mouth.y * h)
        )

        right_mouth_pt = (
            int(right_mouth.x * w),
            int(right_mouth.y * h)
        )

        left_eye_points  = []
        right_eye_points = []

        for idx in LEFT_EYE:
            lm = face.landmark[idx]
            left_eye_points.append((int(lm.x * w), int(lm.y * h)))

        for idx in RIGHT_EYE:
            lm = face.landmark[idx]
            right_eye_points.append((int(lm.x * w), int(lm.y * h)))
        cv2.line(
            frame,
            top_mouth,
            bottom_mouth,
            COLOR_ACCENT_CYAN,
            2,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            left_mouth_pt,
            right_mouth_pt,
            COLOR_MUTED,
            1,
            cv2.LINE_AA
        )

        for pt in [
            top_mouth,
            bottom_mouth,
            left_mouth_pt,
            right_mouth_pt
        ]:
            cv2.circle(
                frame,
                pt,
                4,
                COLOR_ACCENT_AMBER,
                -1,
                cv2.LINE_AA
            )

        if len(left_eye_points) == 6 and len(right_eye_points) == 6:
            left_ear_val  = calculate_ear(left_eye_points)
            right_ear_val = calculate_ear(right_eye_points)
            ear   = (left_ear_val + right_ear_val) / 2.0
            mar = calculate_mar(
            top_mouth,
            bottom_mouth,
            left_mouth_pt,
            right_mouth_pt  
        )
            # Smooth EAR
            ear_smooth = 0.75 * ear_smooth + 0.25 * ear

            eyes_open = ear >= EAR_THRESHOLD

            # Timer logic
            if not eyes_open:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elapsed  = time.time() - closed_start_time
                progress = min(elapsed / ALARM_DELAY_SEC, 1.0)
            else:
                closed_start_time = None
                progress          = 0.0

            # ---- Draw eye landmarks ----
            eye_line_color = COLOR_ACCENT_GREEN if eyes_open else COLOR_ACCENT_RED

            draw_ear_lines(frame, left_eye_points,  eye_line_color)
            draw_ear_lines(frame, right_eye_points, eye_line_color)

            dot_color = COLOR_ACCENT_CYAN if eyes_open else COLOR_ACCENT_AMBER
            draw_eye_landmark(frame, left_eye_points,  dot_color, eye_line_color, radius=3)
            draw_eye_landmark(frame, right_eye_points, dot_color, eye_line_color, radius=3)

            # Alarm trigger
            if not eyes_open and elapsed >= ALARM_DELAY_SEC:
                if not alarm_playing:
                    try:
                        pygame.mixer.music.play(-1)
                    except:
                        pass
                    alarm_playing = True
            else:
                if eyes_open and alarm_playing:
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass
                    alarm_playing = False

    # ============================================================
    # UI OVERLAY
    # ============================================================

    PANEL_W  = 210
    PANEL_H  = 260
    PANEL_X  = 16
    PANEL_Y  = 16
    MARGIN   = 14
    LINE_H   = 26

    # -- Main panel --
    draw_panel(frame, PANEL_X, PANEL_Y, PANEL_W, PANEL_H)

    # Title bar
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (PANEL_X, PANEL_Y),
                  (PANEL_X + PANEL_W, PANEL_Y + 28),
                  (40, 38, 34), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Title
    cv2.putText(frame, "EYE ALERT SYSTEM",
                (PANEL_X + MARGIN, PANEL_Y + 19),
                FONT, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame,
             (PANEL_X + 1, PANEL_Y + 28),
             (PANEL_X + PANEL_W - 1, PANEL_Y + 28),
             (55, 52, 46), 1)

    top = PANEL_Y + 44

    # -- STATUS row --
    put_label(frame, "STATUS", PANEL_X + MARGIN, top, COLOR_MUTED)
    top += 18

    if not face_detected:
        status_text  = "NO FACE"
        status_color = COLOR_MUTED
    elif eyes_open:
        status_text  = "AWAKE"
        status_color = COLOR_ACCENT_GREEN
    else:
        status_text  = "DROWSY"
        status_color = COLOR_ACCENT_RED

    cv2.putText(frame, status_text,
                (PANEL_X + MARGIN, top + 14),
                FONT_MONO, 0.7, status_color, 2, cv2.LINE_AA)

    # Blinking dot indicator
    dot_x = PANEL_X + PANEL_W - MARGIN - 10
    dot_y = top + 7
    dot_c = status_color if (int(time.time() * 2) % 2 == 0 or eyes_open) else COLOR_PANEL_BG
    cv2.circle(frame, (dot_x, dot_y), 5, dot_c, -1, cv2.LINE_AA)

    top += LINE_H + 10

    # Divider
    cv2.line(frame,
             (PANEL_X + MARGIN, top),
             (PANEL_X + PANEL_W - MARGIN, top),
             (50, 48, 44), 1)
    top += 12

    # -- EAR row --
    put_label(frame, "EAR", PANEL_X + MARGIN, top, COLOR_MUTED)

    # EAR value
    ear_str = f"{ear_smooth:.3f}"
    cv2.putText(frame, ear_str,
                (PANEL_X + PANEL_W - MARGIN - 60, top),
                FONT_MONO, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)

    top += 10

    # EAR bar
    BAR_X = PANEL_X + MARGIN
    BAR_W = PANEL_W - MARGIN * 2
    BAR_H = 6
    BAR_Y = top

    # Background
    cv2.rectangle(frame,
                  (BAR_X, BAR_Y),
                  (BAR_X + BAR_W, BAR_Y + BAR_H),
                  (45, 43, 38), -1)

    # Threshold marker
    thresh_x = BAR_X + int(BAR_W * min(EAR_THRESHOLD / 0.4, 1.0))
    cv2.line(frame,
             (thresh_x, BAR_Y - 2),
             (thresh_x, BAR_Y + BAR_H + 2),
             COLOR_MUTED, 1)

    # Fill
    ear_fill  = min(ear_smooth / 0.40, 1.0)
    fill_w    = int(BAR_W * ear_fill)
    bar_color = COLOR_ACCENT_GREEN if eyes_open else COLOR_ACCENT_RED
    if fill_w > 0:
        cv2.rectangle(frame,
                      (BAR_X, BAR_Y),
                      (BAR_X + fill_w, BAR_Y + BAR_H),
                      bar_color, -1)

    top += BAR_H + 18

    # ==================================
    # MAR SECTION
    # ==================================

    cv2.line(frame,
            (PANEL_X + MARGIN, top),
            (PANEL_X + PANEL_W - MARGIN, top),
            (50, 48, 44), 1)

    top += 12

    put_label(frame, "MAR", PANEL_X + MARGIN, top, COLOR_MUTED)

    cv2.putText(
        frame,
        f"{mar:.3f}",
        (PANEL_X + PANEL_W - MARGIN - 60, top),
        FONT_MONO,
        0.52,
        COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    top += 10

    # MAR BAR

    cv2.rectangle(
        frame,
        (BAR_X, top),
        (BAR_X + BAR_W, top + BAR_H),
        (45, 43, 38),
        -1
    )

    mar_fill = min(mar / 1.0, 1.0)

    mar_color = COLOR_ACCENT_GREEN

    if mar > 0.40:
        mar_color = COLOR_ACCENT_AMBER

    if mar > MAR_THRESHOLD:
        mar_color = COLOR_ACCENT_RED

    cv2.rectangle(
        frame,
        (BAR_X, top),
        (BAR_X + int(BAR_W * mar_fill), top + BAR_H),
        mar_color,
        -1
    )

    top += BAR_H + 18

    # ==================================
    # ALERT TIMER SECTION
    # ==================================

    cv2.line(frame,
            (PANEL_X + MARGIN, top),
            (PANEL_X + PANEL_W - MARGIN, top),
            (50, 48, 44), 1)

    top += 12

    put_label(frame, "ALERT IN", PANEL_X + MARGIN, top, COLOR_MUTED)

    if not eyes_open and closed_start_time is not None:
        remaining = max(0.0, ALARM_DELAY_SEC - elapsed)
        timer_str = f"{remaining:.1f}s"
        t_color   = lerp_color(COLOR_ACCENT_GREEN, COLOR_ACCENT_RED, progress)
    else:
        timer_str = f"{ALARM_DELAY_SEC:.1f}s"
        t_color   = COLOR_MUTED

    cv2.putText(frame, timer_str,
                (PANEL_X + PANEL_W - MARGIN - 45, top),
                FONT_MONO, 0.52, t_color, 1, cv2.LINE_AA)

    top += 10

    # Progress bar (ALERT)
    BAR_Y2 = top
    cv2.rectangle(frame,
                  (BAR_X, BAR_Y2),
                  (BAR_X + BAR_W, BAR_Y2 + BAR_H),
                  (45, 43, 38), -1)

    if progress > 0:
        prog_color = lerp_color(COLOR_ACCENT_GREEN, COLOR_ACCENT_RED, progress)
        cv2.rectangle(frame,
                      (BAR_X, BAR_Y2),
                      (BAR_X + int(BAR_W * progress), BAR_Y2 + BAR_H),
                      prog_color, -1)

    # ============================================================
    # ALARM BANNER (full-width bottom strip)
    # ============================================================

    if alarm_playing:
        flash = int(time.time() * 4) % 2 == 0
        if flash:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 54), (w, h), COLOR_ACCENT_RED, -1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        banner_text = "!  DROWSINESS DETECTED  —  STAY ALERT  !"
        (bw, _), _  = cv2.getTextSize(banner_text, FONT, 0.6, 2)
        cv2.putText(frame, banner_text,
                    ((w - bw) // 2, h - 22),
                    FONT, 0.6, COLOR_ACCENT_RED if flash else COLOR_ACCENT_AMBER,
                    2, cv2.LINE_AA)

        # Side bars
        cv2.rectangle(frame, (0, h - 54), (4, h), COLOR_ACCENT_RED, -1)
        cv2.rectangle(frame, (w - 4, h - 54), (w, h), COLOR_ACCENT_RED, -1)

    # ============================================================
    # ARC PROGRESS (bottom-right corner)
    # ============================================================

    if not eyes_open and progress > 0:
        arc_cx = w - 54
        arc_cy = h - 70
        arc_r  = 32
        arc_col = lerp_color(COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, progress)
        draw_arc_progress(frame, arc_cx, arc_cy, arc_r, progress, arc_col, thickness=5)

        pct_str = f"{int(progress * 100)}%"
        put_text_centered(frame, pct_str, arc_cx, arc_cy + 5,
                          FONT_MONO, 0.45, arc_col, 1)

    # ============================================================
    # TOP-RIGHT: face / no-face badge
    # ============================================================

    badge_label = "FACE DETECTED" if face_detected else "NO FACE FOUND"
    badge_color = COLOR_ACCENT_CYAN if face_detected else COLOR_MUTED
    (bw2, _), _ = cv2.getTextSize(badge_label, FONT, 0.38, 1)
    bx = w - bw2 - MARGIN * 2 - 8
    by = 16
    draw_panel(frame, bx, by, bw2 + 18, 22, alpha=1.0)
    cv2.putText(frame, badge_label,
                (bx + 8, by + 15),
                FONT, 0.38, badge_color, 1, cv2.LINE_AA)

    # ============================================================
    # DISPLAY
    # ============================================================

    cv2.imshow("Eye Alert System", frame)
    cv2.imshow("Eye Alert System", frame)
    cv2.setWindowProperty("Eye Alert System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("Eye Alert System", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
try:
    pygame.mixer.music.stop()
    pygame.quit()
except:
    pass
cv2.destroyAllWindows()