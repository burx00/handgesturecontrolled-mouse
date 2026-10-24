import cv2
import time
import math
from pathlib import Path

import numpy as np
import mediapipe as mp
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from Quartz import (
    CGEventCreateMouseEvent,
    CGEventCreateScrollWheelEvent,
    CGEventPost,
    CGEventSetIntegerValueField,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
    kCGHIDEventTap,
    kCGMouseEventClickState,
    kCGScrollEventUnitPixel,
)


# ==========================================
# AI VIRTUAL MOUSE + FACE EXPRESSIONS
# MACOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL = BASE_DIR / "hand_landmarker.task"
FACE_MODEL = BASE_DIR / "face_landmarker.task"

for model in [HAND_MODEL, FACE_MODEL]:
    if not model.exists():
        raise FileNotFoundError(f"Missing model: {model.name}")

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480

# Cursor
ACTIVE_MARGIN = 0.15
SMOOTH_MIN = 0.12      # smoothing when moving slowly (steady cursor)
SMOOTH_MAX = 0.55      # smoothing when moving fast (low lag)

# Pinch (ratios relative to palm width)
PINCH_ON = 0.38
PINCH_OFF = 0.58

# Click / drag
DRAG_HOLD_TIME = 0.30          # hold pinch this long -> drag starts
DRAG_MOVE_PX = 40              # or move this far while pinching -> drag starts
DOUBLE_CLICK_TIME = 0.40
DOUBLE_CLICK_DIST = 30
RIGHT_CLICK_COOLDOWN = 0.60
NO_HAND_RELEASE_FRAMES = 10    # release dragged item if hand lost this long

# Scroll
SCROLL_SPEED = 2200            # higher = faster scrolling
SCROLL_DEADZONE = 0.003
SCROLL_INVERT = False          # set True if direction feels backwards

# Facial landmark measurement thresholds.
SMILE_RATIO_THRESHOLD = 3.5
MOUTH_OPEN_THRESHOLD = 0.20
EYE_CLOSED_THRESHOLD = 0.17

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


# ==========================================
# LOW-LEVEL MOUSE (Quartz = real drag events)
# ==========================================

def mouse_move(x, y):
    e = CGEventCreateMouseEvent(
        None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft
    )
    CGEventPost(kCGHIDEventTap, e)


def mouse_down(x, y, click_state=1):
    e = CGEventCreateMouseEvent(
        None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft
    )
    CGEventSetIntegerValueField(e, kCGMouseEventClickState, click_state)
    CGEventPost(kCGHIDEventTap, e)


def mouse_up(x, y, click_state=1):
    e = CGEventCreateMouseEvent(
        None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft
    )
    CGEventSetIntegerValueField(e, kCGMouseEventClickState, click_state)
    CGEventPost(kCGHIDEventTap, e)


def mouse_drag(x, y):
    e = CGEventCreateMouseEvent(
        None, kCGEventLeftMouseDragged, (x, y), kCGMouseButtonLeft
    )
    CGEventPost(kCGHIDEventTap, e)


def right_click(x, y):
    d = CGEventCreateMouseEvent(
        None, kCGEventRightMouseDown, (x, y), kCGMouseButtonRight
    )
    u = CGEventCreateMouseEvent(
        None, kCGEventRightMouseUp, (x, y), kCGMouseButtonRight
    )
    CGEventPost(kCGHIDEventTap, d)
    CGEventPost(kCGHIDEventTap, u)


def scroll_pixels(amount):
    if amount == 0:
        return
    e = CGEventCreateScrollWheelEvent(
        None, kCGScrollEventUnitPixel, 1, int(amount)
    )
    CGEventPost(kCGHIDEventTap, e)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def normalized_pinch(landmarks, finger_tip):
    """Thumb tip (4) to given fingertip, relative to palm width."""
    palm_width = max(distance(landmarks[5], landmarks[17]), 0.001)
    return distance(landmarks[4], landmarks[finger_tip]) / palm_width


def finger_extended(landmarks, tip, pip):
    wrist = landmarks[0]
    return distance(landmarks[tip], wrist) > distance(landmarks[pip], wrist) * 1.10


def is_scroll_pose(landmarks):
    """Index + middle up, ring + pinky folded."""
    return (
        finger_extended(landmarks, 8, 6)
        and finger_extended(landmarks, 12, 10)
        and not finger_extended(landmarks, 16, 14)
        and not finger_extended(landmarks, 20, 18)
    )


def cursor_target(landmarks):
    """
    Use the knuckles (index + middle MCP) instead of the fingertip,
    so the cursor does NOT jump when you pinch.
    """
    x = (landmarks[5].x + landmarks[9].x) / 2
    y = (landmarks[5].y + landmarks[9].y) / 2

    sx = np.interp(x, [ACTIVE_MARGIN, 1 - ACTIVE_MARGIN], [5, SCREEN_WIDTH - 6])
    sy = np.interp(y, [ACTIVE_MARGIN, 1 - ACTIVE_MARGIN], [5, SCREEN_HEIGHT - 6])
    return float(sx), float(sy)


def pixel_point(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


# Soft colour palette (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
C_TEXT = (245, 245, 245)
C_DIM = (175, 170, 165)
C_ACCENT = (255, 205, 110)     # soft sky blue
C_GOOD = (150, 230, 150)       # mint green
C_WARN = (90, 175, 255)        # warm orange
C_PINK = (210, 150, 255)       # soft pink
C_PANEL = (30, 27, 25)         # dark glass panel


def draw_text(frame, message, x, y, color=C_TEXT, scale=0.55):
    pos = (int(x), int(y))
    cv2.putText(frame, str(message), (pos[0] + 1, pos[1] + 1),
                FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(message), pos, FONT, scale, color, 1, cv2.LINE_AA)


def text_width(message, scale):
    return cv2.getTextSize(str(message), FONT, scale, 1)[0][0]


def draw_text_right(frame, message, x_right, y, color=C_DIM, scale=0.45):
    draw_text(frame, message, x_right - text_width(message, scale), y, color, scale)


def rounded_panel(frame, x1, y1, x2, y2, color=C_PANEL, alpha=0.55, radius=14):
    # translucent rounded rectangle ("glass" panel)
    overlay = frame.copy()
    r = radius
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def status_row(frame, label, value, x, y, color):
    cv2.circle(frame, (x + 4, y - 5), 4, color, -1, cv2.LINE_AA)
    draw_text(frame, label, x + 18, y, C_DIM, 0.45)
    draw_text(frame, value, x + 82, y, color, 0.5)


def glow_dot(frame, center, color, radius=7):
    layer = frame.copy()
    cv2.circle(layer, center, radius * 2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.28, frame, 0.72, 0, frame)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, center, max(radius // 2, 2), (255, 255, 255), -1, cv2.LINE_AA)


# ==========================================
# FACE EXPRESSION DETECTION (measurements only, nothing is drawn on the face)
# ==========================================

def eye_aspect_ratio(face, left, right, top, bottom):
    horizontal = max(distance(face[left], face[right]), 0.001)
    vertical = distance(face[top], face[bottom])
    return vertical / horizontal


def analyze_face(face):
    mouth_width = distance(face[61], face[291])
    mouth_height = max(distance(face[13], face[14]), 0.001)
    smile_ratio = mouth_width / mouth_height

    face_width = max(distance(face[234], face[454]), 0.001)
    mouth_open_ratio = mouth_height / face_width

    left_eye = eye_aspect_ratio(face, 33, 133, 159, 145)
    right_eye = eye_aspect_ratio(face, 362, 263, 386, 374)
    eyes_average = (left_eye + right_eye) / 2

    blinking = eyes_average < EYE_CLOSED_THRESHOLD

    if mouth_open_ratio > MOUTH_OPEN_THRESHOLD:
        expression = "MOUTH OPEN / SURPRISED-LIKE"
    elif smile_ratio > SMILE_RATIO_THRESHOLD and mouth_width / face_width > 0.34:
        expression = "SMILING"
    else:
        expression = "NEUTRAL EXPRESSION"

    return {
        "expression": expression,
        "blink": blinking,
        "smile_ratio": smile_ratio,
        "mouth_open": mouth_open_ratio,
        "eye_ratio": eyes_average,
    }


# ==========================================
# HAND OVERLAY
# ==========================================

HAND_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_hand(frame, hand, pinching):
    height, width = frame.shape[:2]
    pts = [pixel_point(p, width, height) for p in hand]

    layer = frame.copy()
    for a, b in HAND_LINES:
        cv2.line(layer, pts[a], pts[b], C_TEXT, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(layer, p, 3, C_TEXT, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.45, frame, 0.55, 0, frame)

    color = C_GOOD if pinching else C_ACCENT
    cv2.line(frame, pts[4], pts[8], color, 2, cv2.LINE_AA)
    glow_dot(frame, pts[4], color)
    glow_dot(frame, pts[8], color)


# ==========================================
# MEDIAPIPE MODEL OPTIONS
# ==========================================

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError("Cannot open MacBook camera.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# ==========================================
# CONTROLLER STATE
# ==========================================

enabled = False

cursor_x, cursor_y = pyautogui.position()
cursor_x = float(cursor_x)
cursor_y = float(cursor_y)

pinch_state = "IDLE"          # IDLE / PENDING / DRAGGING
pinch_start = 0.0
anchor_x, anchor_y = 0.0, 0.0

last_click_time = 0.0
last_click_pos = (0.0, 0.0)
last_click_count = 0

right_active = False
last_right_click = 0.0

scroll_frames = 0
prev_scroll_y = None
scroll_accum = 0.0

no_hand_frames = 0

start_time = time.monotonic()
last_timestamp = -1

face_status = "NO FACE"
gesture_status = "PAUSED"


def do_click(x, y, now):
    """Left click with proper double-click counting."""
    global last_click_time, last_click_pos, last_click_count

    close = math.hypot(x - last_click_pos[0], y - last_click_pos[1]) < DOUBLE_CLICK_DIST
    if now - last_click_time < DOUBLE_CLICK_TIME and close:
        last_click_count = min(last_click_count + 1, 3)
    else:
        last_click_count = 1

    mouse_down(x, y, last_click_count)
    mouse_up(x, y, last_click_count)

    last_click_time = now
    last_click_pos = (x, y)
    return "DOUBLE CLICK" if last_click_count == 2 else "LEFT CLICK"


def release_drag_if_needed():
    global pinch_state
    if pinch_state == "DRAGGING":
        mouse_up(int(cursor_x), int(cursor_y))
    pinch_state = "IDLE"


# ==========================================
# MAIN APPLICATION
# ==========================================

try:
    with (
        vision.HandLandmarker.create_from_options(hand_options) as hand_detector,
        vision.FaceLandmarker.create_from_options(face_options) as face_detector,
    ):

        while True:
            success, frame = camera.read()

            if not success:
                print("Camera read failed.")
                break

            pyautogui.failSafeCheck()

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )

            timestamp = int((time.monotonic() - start_time) * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            hands = hand_detector.detect_for_video(mp_image, timestamp)
            faces = face_detector.detect_for_video(mp_image, timestamp)

            # ------------------------------
            # FACE EXPRESSIONS (no drawing on the face)
            # ------------------------------

            face_status = "NO FACE"
            blink_status = "NO FACE"
            metrics = None

            if faces.face_landmarks:
                metrics = analyze_face(faces.face_landmarks[0])
                face_status = metrics["expression"]
                blink_status = (
                    "EYES CLOSED / BLINK" if metrics["blink"] else "EYES OPEN"
                )

            # ------------------------------
            # HAND GESTURES
            # ------------------------------

            gesture_status = "ACTIVE - NO HAND" if enabled else "PAUSED"

            if hands.hand_landmarks:
                no_hand_frames = 0
                hand = hands.hand_landmarks[0]

                draw_hand(frame, hand, normalized_pinch(hand, 8) < PINCH_ON)

                if enabled:
                    now = time.monotonic()

                    pinch_ratio = normalized_pinch(hand, 8)     # thumb + index
                    right_ratio = normalized_pinch(hand, 12)    # thumb + middle

                    target_x, target_y = cursor_target(hand)

                    # ---------- scroll detection (with hysteresis) ----------
                    if is_scroll_pose(hand) and pinch_state == "IDLE":
                        scroll_frames = min(scroll_frames + 1, 6)
                    else:
                        scroll_frames = max(scroll_frames - 1, 0)

                    scrolling = scroll_frames >= 3 and pinch_state == "IDLE"

                    if scrolling:
                        ref_y = (hand[8].y + hand[12].y) / 2

                        if prev_scroll_y is not None:
                            dy = ref_y - prev_scroll_y
                            if abs(dy) > SCROLL_DEADZONE:
                                amount = -dy * SCROLL_SPEED
                                if SCROLL_INVERT:
                                    amount = -amount
                                scroll_accum += amount
                                whole = int(scroll_accum)
                                scroll_accum -= whole
                                scroll_pixels(whole)

                        prev_scroll_y = ref_y
                        gesture_status = "SCROLLING"

                    else:
                        prev_scroll_y = None
                        scroll_accum = 0.0

                        # ---------- cursor smoothing (adaptive) ----------
                        if pinch_state != "PENDING":
                            d = math.hypot(target_x - cursor_x, target_y - cursor_y)
                            alpha = SMOOTH_MIN + (SMOOTH_MAX - SMOOTH_MIN) * min(d / 250, 1.0)
                            cursor_x += (target_x - cursor_x) * alpha
                            cursor_y += (target_y - cursor_y) * alpha

                        safe_x = int(np.clip(cursor_x, 5, SCREEN_WIDTH - 6))
                        safe_y = int(np.clip(cursor_y, 5, SCREEN_HEIGHT - 6))

                        # ---------- pinch state machine ----------
                        if pinch_state == "IDLE":
                            mouse_move(safe_x, safe_y)
                            gesture_status = "MOVING CURSOR"

                            if pinch_ratio < PINCH_ON:
                                pinch_state = "PENDING"
                                pinch_start = now
                                anchor_x, anchor_y = safe_x, safe_y

                            # right click: thumb + middle (index not pinching)
                            elif (
                                right_ratio < PINCH_ON
                                and pinch_ratio > PINCH_ON + 0.12
                                and not right_active
                                and now - last_right_click > RIGHT_CLICK_COOLDOWN
                            ):
                                right_active = True
                                last_right_click = now
                                right_click(safe_x, safe_y)
                                gesture_status = "RIGHT CLICK"

                        elif pinch_state == "PENDING":
                            gesture_status = "PINCH..."

                            moved = math.hypot(
                                target_x - anchor_x, target_y - anchor_y
                            )

                            if pinch_ratio > PINCH_OFF:
                                # quick pinch + release = click
                                gesture_status = do_click(anchor_x, anchor_y, now)
                                pinch_state = "IDLE"

                            elif (
                                now - pinch_start > DRAG_HOLD_TIME
                                or moved > DRAG_MOVE_PX
                            ):
                                # held or moved = start drag
                                mouse_down(int(anchor_x), int(anchor_y), 1)
                                pinch_state = "DRAGGING"
                                cursor_x, cursor_y = anchor_x, anchor_y
                                gesture_status = "DRAGGING"

                        elif pinch_state == "DRAGGING":
                            mouse_drag(safe_x, safe_y)
                            gesture_status = "DRAGGING"

                            if pinch_ratio > PINCH_OFF:
                                mouse_up(safe_x, safe_y, 1)
                                pinch_state = "IDLE"
                                gesture_status = "DROPPED"

                    if right_ratio > PINCH_OFF:
                        right_active = False

                else:
                    pinch_state = "IDLE"

            else:
                # hand lost
                no_hand_frames += 1
                scroll_frames = 0
                prev_scroll_y = None

                if pinch_state == "PENDING":
                    pinch_state = "IDLE"
                elif (
                    pinch_state == "DRAGGING"
                    and no_hand_frames >= NO_HAND_RELEASE_FRAMES
                ):
                    release_drag_if_needed()

            # ------------------------------
            # DRAW DASHBOARD
            # ------------------------------

            # top panel
            rounded_panel(frame, 12, 12, width - 12, 124)
            draw_text(frame, "AI VIRTUAL MOUSE", 28, 38, C_TEXT, 0.62)
            draw_text_right(frame, "hands + expressions", width - 28, 38, C_DIM, 0.45)

            status_row(frame, "MOUSE", gesture_status, 28, 66,
                       C_GOOD if enabled else C_WARN)
            status_row(frame, "FACE", face_status, 28, 88, C_PINK)
            status_row(frame, "EYES", blink_status, 28, 110, C_ACCENT)

            if metrics is not None:
                draw_text_right(frame, f"smile {metrics['smile_ratio']:.2f}",
                                width - 28, 88, C_DIM, 0.42)
                draw_text_right(frame, f"mouth {metrics['mouth_open']:.2f}",
                                width - 28, 110, C_DIM, 0.42)

            # bottom panel
            rounded_panel(frame, 12, height - 62, width - 12, height - 12)
            draw_text(
                frame,
                "Pinch: click | Pinch+move: drag | 2 fingers: scroll | Thumb+middle: right",
                28, height - 40, C_DIM, 0.40,
            )
            draw_text(
                frame, "S start    P pause    Q quit",
                28, height - 21, C_TEXT, 0.47,
            )

            cv2.imshow("AI Mouse + Facial Expressions", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                enabled = True
                cursor_x, cursor_y = pyautogui.position()
                cursor_x = float(cursor_x)
                cursor_y = float(cursor_y)
                pinch_state = "IDLE"
                print("Mouse activated.")

            elif key == ord("p"):
                release_drag_if_needed()
                enabled = False
                print("Mouse paused.")

            elif key == ord("q"):
                break

except pyautogui.FailSafeException:
    print("Emergency mouse stop activated.")

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    release_drag_if_needed()
    camera.release()
    cv2.destroyAllWindows()
    print("Application closed.")