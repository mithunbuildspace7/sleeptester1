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
    eye_alarm  = pygame.mixer.Sound("alarm.wav")
    yawn_alarm = pygame.mixer.Sound("silly.wav")
except:
    pass

# -----------------------------
# DESIGN TOKENS
# -----------------------------

COLOR_BG_DARK      = (15, 15, 20)
COLOR_ACCENT_CYAN  = (220, 200, 50)
COLOR_ACCENT_AMBER = (0, 165, 245)
COLOR_ACCENT_RED   = (50, 50, 230)
COLOR_ACCENT_GREEN = (100, 210, 80)
COLOR_WHITE        = (240, 240, 240)
COLOR_MUTED        = (120, 118, 110)
COLOR_PANEL_BG     = (28, 26, 22)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
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
    vertical   = distance(top, bottom)
    horizontal = distance(left, right)
    return vertical / horizontal

def calculate_yaw(face, w, h):
    nose      = face.landmark[1]
    left_eye  = face.landmark[33]
    right_eye = face.landmark[263]

    nose_x      = nose.x * w
    left_eye_x  = left_eye.x * w
    right_eye_x = right_eye.x * w
    eye_center_x = (left_eye_x + right_eye_x) / 2.0
    eye_width    = abs(right_eye_x - left_eye_x)
    if eye_width == 0:
        return 0.0
    offset  = (nose_x - eye_center_x) / eye_width
    yaw_deg = offset * 90.0
    return yaw_deg

def get_best_ear(left_ear, right_ear, yaw_deg):
    YAW_LIMIT = 28
    if yaw_deg > YAW_LIMIT:
        return left_ear, "LEFT"
    elif yaw_deg < -YAW_LIMIT:
        return right_ear, "RIGHT"
    else:
        return (left_ear + right_ear) / 2.0, "BOTH"

def draw_panel(frame, x, y, w, h, alpha=0.75):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 48, 44), 1)

def draw_eye_landmark(frame, points, color, line_color, radius=4):
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        cv2.line(frame, p1, p2, line_color, 1, cv2.LINE_AA)
    for pt in points:
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

def draw_ear_lines(frame, pts, color):
    cv2.line(frame, pts[1], pts[5], color, 1, cv2.LINE_AA)
    cv2.line(frame, pts[2], pts[4], color, 1, cv2.LINE_AA)
    cv2.line(frame, pts[0], pts[3], COLOR_MUTED, 1, cv2.LINE_AA)

def draw_arc_progress(frame, cx, cy, radius, progress, color, thickness=6):
    start_angle = -90
    end_angle   = start_angle + int(360 * progress)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                0, 360, (50, 48, 44), thickness, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                    start_angle, end_angle, color, thickness, cv2.LINE_AA)

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def put_text_centered(frame, text, cx, y, font, scale, color, thickness=1):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, (cx - tw // 2, y), font, scale, color, thickness, cv2.LINE_AA)

def put_label(frame, text, x, y, color, scale=0.45, thickness=1):
    cv2.putText(frame, text.upper(), (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

# -------------------------------------------------------
# LETTERBOX — fits the camera frame into the display
# canvas without stretching, with black bars if needed
# -------------------------------------------------------
def letterbox_frame(frame, target_w, target_h):
    fh, fw = frame.shape[:2]
    scale  = min(target_w / fw, target_h / fh)
    new_w  = int(fw * scale)
    new_h  = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off   = (target_w - new_w) // 2
    y_off   = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

# -----------------------------
# FACEMESH SETUP
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE    = [33,  160, 158, 133, 153, 144]
RIGHT_EYE   = [362, 385, 387, 263, 373, 380]
UPPER_LIP   = 13
LOWER_LIP   = 14
LEFT_MOUTH  = 78
RIGHT_MOUTH = 308

# -----------------------------
# SETTINGS
# -----------------------------

EAR_THRESHOLD   = 0.2
ALARM_DELAY_SEC = 1.5
YAW_LIMIT       = 28

# -------------------------------------------------------
# FIX 1 — MAR threshold raised from 0.1 → 0.50
#   Talking opens the mouth ~0.1–0.3; a real yawn is 0.5+
# -------------------------------------------------------
MAR_THRESHOLD = 0.50

# -------------------------------------------------------
# FIX 2 — Minimum open duration before a yawn is counted
#   Mouth must stay above MAR_THRESHOLD for at least this
#   many seconds. Talking spikes are short (<0.3 s).
#   A real yawn lasts 1–4 seconds.
# -------------------------------------------------------
YAWN_MIN_DURATION = 1.0     # seconds mouth must stay open

# -------------------------------------------------------
# FIX 3 — Cooldown between consecutive yawn registrations
#   Prevents a single slow open-close from registering
#   multiple times due to oscillation near the threshold.
# -------------------------------------------------------
YAWN_COOLDOWN_SEC = 1    # seconds to ignore after a yawn is logged

# Yawn window / counter
YAWN_WINDOW_SEC  = 60
YAWN_COUNT_LIMIT = 3

# -------------------------------------------------------
# YAWN STATE — expanded to support duration + cooldown
# -------------------------------------------------------
yawn_timestamps   = []          # times of completed yawn events
currently_yawning = False       # True while mouth is open above threshold
yawn_open_start   = None        # when the current mouth-open began
last_yawn_time    = 0.0         # timestamp of last logged yawn (for cooldown)
yawn_alarm_played = False       # so alarm only plays once per 3-yawn trigger

# -------------------------------------------------------
# FIX 4 — Banner timer: show the yawn warning banner for
#   N seconds after the alarm fires, then hide it.
#   Without this the banner disappears the instant
#   timestamps are cleared.
# -------------------------------------------------------
YAWN_BANNER_DURATION = 6.0      # seconds to show the yawn banner
yawn_banner_until    = 0.0      # absolute time when banner should hide

# EYE CLOSURE STATE
closed_start_time = None
alarm_playing     = False

# Display values
ear_smooth  = 0.25
mar         = 0.0
yaw_deg     = 0.0
eye_mode    = "BOTH"

# -----------------------------
# WINDOW SETUP
# -------------------------------------------------------
# FIX 5 — Instead of fullscreen (which stretches the feed
#   to fill any monitor aspect ratio), open a normal
#   resizable window at a sensible default size.
#   The letterbox_frame() function keeps the camera image
#   at its native aspect ratio with black bars if needed.
# -------------------------------------------------------
DISPLAY_W = 1280
DISPLAY_H = 720

cv2.namedWindow("Eye Alert System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye Alert System", DISPLAY_W, DISPLAY_H)

# -----------------------------
# WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

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

        yaw_deg = calculate_yaw(face, w, h)

        upper_lip  = face.landmark[UPPER_LIP]
        lower_lip  = face.landmark[LOWER_LIP]
        left_mouth = face.landmark[LEFT_MOUTH]
        right_mouth= face.landmark[RIGHT_MOUTH]

        top_mouth    = (int(upper_lip.x * w),    int(upper_lip.y * h))
        bottom_mouth = (int(lower_lip.x * w),    int(lower_lip.y * h))
        left_mouth_pt  = (int(left_mouth.x * w),  int(left_mouth.y * h))
        right_mouth_pt = (int(right_mouth.x * w), int(right_mouth.y * h))

        cv2.line(frame, top_mouth, bottom_mouth, COLOR_ACCENT_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, left_mouth_pt, right_mouth_pt, COLOR_MUTED, 1, cv2.LINE_AA)
        for pt in [top_mouth, bottom_mouth, left_mouth_pt, right_mouth_pt]:
            cv2.circle(frame, pt, 4, COLOR_ACCENT_AMBER, -1, cv2.LINE_AA)

        left_eye_points  = []
        right_eye_points = []
        for idx in LEFT_EYE:
            lm = face.landmark[idx]
            left_eye_points.append((int(lm.x * w), int(lm.y * h)))
        for idx in RIGHT_EYE:
            lm = face.landmark[idx]
            right_eye_points.append((int(lm.x * w), int(lm.y * h)))

        if len(left_eye_points) == 6 and len(right_eye_points) == 6:
            left_ear_val  = calculate_ear(left_eye_points)
            right_ear_val = calculate_ear(right_eye_points)

            ear, eye_mode = get_best_ear(left_ear_val, right_ear_val, yaw_deg)
            mar           = calculate_mar(top_mouth, bottom_mouth,
                                          left_mouth_pt, right_mouth_pt)

            ear_smooth = 0.75 * ear_smooth + 0.25 * ear
            eyes_open  = ear >= EAR_THRESHOLD

            # Eye closure timer
            if not eyes_open:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elapsed  = time.time() - closed_start_time
                progress = min(elapsed / ALARM_DELAY_SEC, 1.0)
            else:
                closed_start_time = None
                progress = 0.0

            eye_line_color = COLOR_ACCENT_GREEN if eyes_open else COLOR_ACCENT_RED
            draw_ear_lines(frame, left_eye_points,  eye_line_color)
            draw_ear_lines(frame, right_eye_points, eye_line_color)
            dot_color = COLOR_ACCENT_CYAN if eyes_open else COLOR_ACCENT_AMBER
            draw_eye_landmark(frame, left_eye_points,  dot_color, eye_line_color, radius=3)
            draw_eye_landmark(frame, right_eye_points, dot_color, eye_line_color, radius=3)

            # ---------------------------------------------------
            # YAWN DETECTION — fixed version
            #
            # A yawn event is logged only when ALL three are true:
            #   1. MAR exceeded the (raised) threshold
            #   2. Mouth stayed open for >= YAWN_MIN_DURATION secs
            #   3. At least YAWN_COOLDOWN_SEC since the last yawn
            # ---------------------------------------------------
            now = time.time()

            if mar > MAR_THRESHOLD:
                # Mouth is open — start or continue timing the opening
                if yawn_open_start is None:
                    yawn_open_start = now
                currently_yawning = True
            else:
                # Mouth just closed (or never reached threshold)
                if currently_yawning and yawn_open_start is not None:
                    open_duration = now - yawn_open_start
                    time_since_last = now - last_yawn_time

                    # Log only if long enough AND cooldown elapsed
                    if (open_duration >= YAWN_MIN_DURATION and
                            time_since_last >= YAWN_COOLDOWN_SEC):
                        yawn_timestamps.append(now)
                        last_yawn_time = now

                # Reset open-state regardless
                yawn_open_start   = None
                currently_yawning = False

            # Prune timestamps outside the 60-second window
            yawn_timestamps = [t for t in yawn_timestamps
                               if now - t <= YAWN_WINDOW_SEC]
            yawn_count = len(yawn_timestamps)

            # ---------------------------------------------------
            # YAWN ALARM — fires once per 3-yawn cycle, then
            # RESETS the counter so it can fire again next cycle
            # ---------------------------------------------------
            if yawn_count >= YAWN_COUNT_LIMIT:
                if not yawn_alarm_played:
                    try:
                        yawn_alarm.play()
                    except:
                        pass
                    yawn_alarm_played = True
                    yawn_banner_until = now + YAWN_BANNER_DURATION

                    # FIX: clear timestamps → counter resets to 0
                    # so the next 3 yawns can trigger another alarm
                    yawn_timestamps.clear()
            else:
                # Allow the alarm to fire again once count resets
                yawn_alarm_played = False

            # Eye closure alarm
            if not eyes_open and elapsed >= ALARM_DELAY_SEC:
                if not alarm_playing:
                    try:
                        eye_alarm.play(-1)
                    except:
                        pass
                    alarm_playing = True
            else:
                if eyes_open and alarm_playing:
                    try:
                        eye_alarm.stop()
                    except:
                        pass
                    alarm_playing = False

    # ============================================================
    # UI OVERLAY
    # ============================================================

    PANEL_W = 210
    PANEL_H = 310
    PANEL_X = 16
    PANEL_Y = 16
    MARGIN  = 14
    LINE_H  = 26

    draw_panel(frame, PANEL_X, PANEL_Y, PANEL_W, PANEL_H)

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (PANEL_X, PANEL_Y),
                  (PANEL_X + PANEL_W, PANEL_Y + 28),
                  (40, 38, 34), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.putText(frame, "EYE ALERT SYSTEM",
                (PANEL_X + MARGIN, PANEL_Y + 19),
                FONT, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)
    cv2.line(frame,
             (PANEL_X + 1, PANEL_Y + 28),
             (PANEL_X + PANEL_W - 1, PANEL_Y + 28),
             (55, 52, 46), 1)

    top = PANEL_Y + 44

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
    dot_x = PANEL_X + PANEL_W - MARGIN - 10
    dot_y = top + 7
    dot_c = status_color if (int(time.time() * 2) % 2 == 0 or eyes_open) else COLOR_PANEL_BG
    cv2.circle(frame, (dot_x, dot_y), 5, dot_c, -1, cv2.LINE_AA)
    top += LINE_H + 10

    cv2.line(frame, (PANEL_X + MARGIN, top), (PANEL_X + PANEL_W - MARGIN, top), (50, 48, 44), 1)
    top += 12

    # EAR
    put_label(frame, "EAR", PANEL_X + MARGIN, top, COLOR_MUTED)
    cv2.putText(frame, f"{ear_smooth:.3f}",
                (PANEL_X + PANEL_W - MARGIN - 60, top),
                FONT_MONO, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    top += 10

    BAR_X = PANEL_X + MARGIN
    BAR_W = PANEL_W - MARGIN * 2
    BAR_H = 6

    cv2.rectangle(frame, (BAR_X, top), (BAR_X + BAR_W, top + BAR_H), (45, 43, 38), -1)
    thresh_x = BAR_X + int(BAR_W * min(EAR_THRESHOLD / 0.4, 1.0))
    cv2.line(frame, (thresh_x, top - 2), (thresh_x, top + BAR_H + 2), COLOR_MUTED, 1)
    ear_fill  = min(ear_smooth / 0.40, 1.0)
    fill_w    = int(BAR_W * ear_fill)
    bar_color = COLOR_ACCENT_GREEN if eyes_open else COLOR_ACCENT_RED
    if fill_w > 0:
        cv2.rectangle(frame, (BAR_X, top), (BAR_X + fill_w, top + BAR_H), bar_color, -1)
    top += BAR_H + 18

    cv2.line(frame, (PANEL_X + MARGIN, top), (PANEL_X + PANEL_W - MARGIN, top), (50, 48, 44), 1)
    top += 12

    # MAR
    put_label(frame, "MAR", PANEL_X + MARGIN, top, COLOR_MUTED)
    cv2.putText(frame, f"{mar:.3f}",
                (PANEL_X + PANEL_W - MARGIN - 60, top),
                FONT_MONO, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    top += 10

    cv2.rectangle(frame, (BAR_X, top), (BAR_X + BAR_W, top + BAR_H), (45, 43, 38), -1)
    mar_fill  = min(mar / 1.0, 1.0)
    mar_color = COLOR_ACCENT_GREEN
    if mar > 0.40:
        mar_color = COLOR_ACCENT_AMBER
    if mar > MAR_THRESHOLD:
        mar_color = COLOR_ACCENT_RED
    cv2.rectangle(frame, (BAR_X, top), (BAR_X + int(BAR_W * mar_fill), top + BAR_H), mar_color, -1)
    top += BAR_H + 18

    cv2.line(frame, (PANEL_X + MARGIN, top), (PANEL_X + PANEL_W - MARGIN, top), (50, 48, 44), 1)
    top += 12

    # YAWN COUNTER
    put_label(frame, "YAWNS / MIN", PANEL_X + MARGIN, top, COLOR_MUTED)
    yawn_count_display = len(yawn_timestamps)
    ycount_color = COLOR_ACCENT_RED if yawn_count_display >= YAWN_COUNT_LIMIT else COLOR_WHITE
    cv2.putText(frame, f"{yawn_count_display}/3",
                (PANEL_X + PANEL_W - MARGIN - 30, top),
                FONT_MONO, 0.48, ycount_color, 1, cv2.LINE_AA)
    top += 12

    dot_spacing = 18
    dot_start_x = PANEL_X + MARGIN
    for i in range(YAWN_COUNT_LIMIT):
        dx = dot_start_x + i * dot_spacing
        dy = top + 6
        if i < yawn_count_display:
            filled_color = (COLOR_ACCENT_RED
                            if yawn_count_display >= YAWN_COUNT_LIMIT
                            else COLOR_ACCENT_AMBER)
            cv2.circle(frame, (dx, dy), 6, filled_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (dx, dy), 7, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (dx, dy), 6, (45, 43, 38), -1, cv2.LINE_AA)
            cv2.circle(frame, (dx, dy), 6, COLOR_MUTED,  1, cv2.LINE_AA)
    top += 20

    cv2.line(frame, (PANEL_X + MARGIN, top), (PANEL_X + PANEL_W - MARGIN, top), (50, 48, 44), 1)
    top += 12

    # ALERT IN
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

    cv2.rectangle(frame, (BAR_X, top), (BAR_X + BAR_W, top + BAR_H), (45, 43, 38), -1)
    if progress > 0:
        prog_color = lerp_color(COLOR_ACCENT_GREEN, COLOR_ACCENT_RED, progress)
        cv2.rectangle(frame, (BAR_X, top),
                      (BAR_X + int(BAR_W * progress), top + BAR_H), prog_color, -1)

    # HEAD POSE BADGE
    badge_label = "FACE DETECTED" if face_detected else "NO FACE FOUND"
    badge_color = COLOR_ACCENT_CYAN if face_detected else COLOR_MUTED
    (bw2, _), _ = cv2.getTextSize(badge_label, FONT, 0.38, 1)
    bx = w - bw2 - MARGIN * 2 - 8
    by = 16
    draw_panel(frame, bx, by, bw2 + 18, 22, alpha=1.0)
    cv2.putText(frame, badge_label, (bx + 8, by + 15),
                FONT, 0.38, badge_color, 1, cv2.LINE_AA)

    if face_detected:
        pose_label = f"YAW  {yaw_deg:+.0f} deg  [{eye_mode}]"
        (pw, _), _ = cv2.getTextSize(pose_label, FONT, 0.35, 1)
        px = w - pw - MARGIN * 2 - 8
        py = by + 28
        abs_yaw = abs(yaw_deg)
        if abs_yaw < YAW_LIMIT:
            pose_color = COLOR_ACCENT_GREEN
        elif abs_yaw < 45:
            pose_color = COLOR_ACCENT_AMBER
        else:
            pose_color = COLOR_ACCENT_RED
        draw_panel(frame, px, py, pw + 18, 20, alpha=0.85)
        cv2.putText(frame, pose_label, (px + 8, py + 14),
                    FONT, 0.35, pose_color, 1, cv2.LINE_AA)

    # EYE ALARM BANNER
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
                    FONT, 0.6,
                    COLOR_ACCENT_RED if flash else COLOR_ACCENT_AMBER,
                    2, cv2.LINE_AA)
        cv2.rectangle(frame, (0,     h - 54), (4, h), COLOR_ACCENT_RED, -1)
        cv2.rectangle(frame, (w - 4, h - 54), (w, h), COLOR_ACCENT_RED, -1)

    # YAWN BANNER — shown for YAWN_BANNER_DURATION seconds after alarm
    if time.time() < yawn_banner_until:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 108), (w, h - 58), COLOR_ACCENT_AMBER, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        yawn_text = "!  3 YAWNS DETECTED  —  TAKE A BREAK  !"
        (yw, _), _ = cv2.getTextSize(yawn_text, FONT, 0.6, 2)
        cv2.putText(frame, yawn_text,
                    ((w - yw) // 2, h - 72),
                    FONT, 0.6, COLOR_ACCENT_AMBER, 2, cv2.LINE_AA)

    # ARC PROGRESS
    if not eyes_open and progress > 0:
        arc_cx  = w - 54
        arc_cy  = h - 70
        arc_r   = 32
        arc_col = lerp_color(COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, progress)
        draw_arc_progress(frame, arc_cx, arc_cy, arc_r, progress, arc_col, thickness=5)
        pct_str = f"{int(progress * 100)}%"
        put_text_centered(frame, pct_str, arc_cx, arc_cy + 5, FONT_MONO, 0.45, arc_col, 1)

    # -------------------------------------------------------
    # FIX 5 — Letterbox the frame into the display canvas
    # so the camera feed keeps its native aspect ratio
    # -------------------------------------------------------
    display = letterbox_frame(frame, DISPLAY_W, DISPLAY_H)
    cv2.imshow("Eye Alert System", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Eye Alert System", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
try:
    pygame.mixer.music.stop()
    pygame.quit()
except:
    pass
cv2.destroyAllWindows()