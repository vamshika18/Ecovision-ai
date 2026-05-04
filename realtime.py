# ============================================================
# realtime.py  –  Stable Live Webcam Waste Detection
# Fix: confidence threshold + temporal smoothing + stable HUD
# ============================================================

import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO
from waste_analysis import analyze_waste

# ─────────────────────────────────────────────
# ⚙️  TUNABLE SETTINGS  (adjust as needed)
# ─────────────────────────────────────────────
MODEL_PATH           = "best (1).pt"
CONFIDENCE_THRESHOLD = 0.50   # ignore detections below this
SMOOTHING_FRAMES     = 8      # how many recent frames to average over
MIN_AGREE_FRAMES     = 3      # label must appear this many frames to show
# ─────────────────────────────────────────────

WINDOW_NAME = "Smart Waste Detection  |  Press Q to quit"
FONT        = cv2.FONT_HERSHEY_SIMPLEX

RISK_COLORS = {
    "Clean":  (0, 210, 90),
    "High":   (30, 30, 220),
    "Medium": (0, 165, 255),
    "Low":    (0, 230, 230),
    "None":   (180, 180, 180),
}

STATUS_COLORS = {
    "Clean Recyclable Waste":     (0, 210, 90),
    "Contaminated / Mixed Waste": (30, 30, 220),
    "Non-Recyclable Waste":       (30, 30, 220),
    "No Waste Detected":          (180, 180, 180),
}


def draw_hud(frame, analysis, fps, stable):
    """Draw semi-transparent info panel onto frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark panel background
    cv2.rectangle(overlay, (5, 5), (w - 5, 148), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    risk_color   = RISK_COLORS.get(analysis["risk_level"], (255, 255, 255))
    status_color = STATUS_COLORS.get(analysis["status"], (255, 255, 255))

    # Stability indicator dot (green = stable, orange = still reading)
    dot_color = (0, 210, 90) if stable else (0, 140, 255)
    cv2.circle(frame, (w - 20, 20), 7, dot_color, -1)
    stab_text = "STABLE" if stable else "READING..."
    cv2.putText(frame, stab_text, (w - 95, 25), FONT, 0.42, dot_color, 1, cv2.LINE_AA)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 95, 48), FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    # Main info lines (shadow pass + colour pass for readability)
    lines = [
        (f"Status       : {analysis['status']}",                                    (10, 32),  status_color,  0.60),
        (f"Risk Level   : {analysis['risk_level']}",                                (10, 58),  risk_color,    0.60),
        (f"Contamination: {analysis['contamination_percent']}%",                    (10, 84),  (0, 200, 255), 0.60),
        (f"Recyclable: {analysis['recyclable_count']}   Non-Recyclable: {analysis['non_recyclable_count']}",
                                                                                    (10, 110), (200, 200, 200), 0.55),
        (f"Tip: {analysis['recommendation'][:60]}",                                 (10, 136), (255, 200, 50), 0.50),
    ]

    for text, pos, color, scale in lines:
        cv2.putText(frame, text, pos, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)   # shadow
        cv2.putText(frame, text, pos, FONT, scale, color,     1, cv2.LINE_AA)   # colour

    return frame


def get_stable_detections(history):
    """
    From the rolling history window, return only class names that
    appeared in >= MIN_AGREE_FRAMES frames (stable), with averaged confidence.
    """
    all_classes = []
    conf_map    = {}

    for frame_classes, frame_confs in history:
        for cls, conf in zip(frame_classes, frame_confs):
            all_classes.append(cls)
            conf_map.setdefault(cls, []).append(conf)

    counts = Counter(all_classes)

    stable_classes = []
    stable_confs   = []
    for cls, count in counts.items():
        if count >= MIN_AGREE_FRAMES:
            stable_classes.append(cls)
            stable_confs.append(sum(conf_map[cls]) / len(conf_map[cls]))

    return stable_classes, stable_confs


def main():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Boost resolution if webcam supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam open. Press Q or ESC to quit.")

    detection_history = deque(maxlen=SMOOTHING_FRAMES)
    last_analysis     = analyze_waste([], [])
    last_stable       = False
    prev_time         = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break

        # FPS
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # YOLO inference — pass conf threshold directly to model
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        frame_classes = []
        frame_confs   = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            frame_classes.append(model.names[cls_id])
            frame_confs.append(conf)

        # Add to rolling window
        detection_history.append((frame_classes, frame_confs))

        # Get temporally stable detections
        stable_classes, stable_confs = get_stable_detections(detection_history)
        is_stable = len(stable_classes) > 0

        if is_stable:
            last_analysis = analyze_waste(stable_classes, stable_confs)
            last_stable   = True
        elif len(detection_history) == SMOOTHING_FRAMES:
            # Buffer full, nothing stable = genuinely empty
            last_analysis = analyze_waste([], [])
            last_stable   = False

        # Draw annotations + HUD
        annotated = results[0].plot()
        annotated = draw_hud(annotated, last_analysis, fps, last_stable)

        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()