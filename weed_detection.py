"""
weed_detection.py
Runs YOLOv8 best.pt model on Pi Cam 3 and displays live detections.

Camera:  0 (left) or 1 (right) — set CAMERA_NUM below
Press 'q' to quit.
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from pathlib import Path

# --- Config ---
CAMERA_NUM   = 0                      # 0 = left cam, 1 = right cam
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CONF_THRESH  = 0.5                    # minimum confidence to show a detection
MODEL_PATH   = Path(__file__).parent / "best.pt"
# --------------

# Colours per class index (BGR)
COLOURS = [
    (0, 255, 0),    # class 0 — green
    (0, 0, 255),    # class 1 — red
    (255, 0, 0),    # class 2 — blue
    (0, 255, 255),  # class 3 — yellow
    (255, 0, 255),  # class 4 — magenta
]


def draw_detections(frame, results):
    """Draw bounding boxes and labels onto the frame."""
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue

            cls_id = int(box.cls[0])
            label  = result.names[cls_id]
            colour = COLOURS[cls_id % len(COLOURS)]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), colour, -1)

            # Label text
            cv2.putText(frame, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return frame


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"Classes: {model.names}")

    print(f"Starting camera {CAMERA_NUM}...")
    cam = Picamera2(camera_num=CAMERA_NUM)
    cam.configure(cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    ))
    cam.start()
    print("Running — press 'q' to quit.")

    try:
        while True:
            frame_rgb = cam.capture_array()

            # Run YOLOv8 inference (model expects RGB, returns results)
            results = model(frame_rgb, verbose=False)

            # Convert to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = draw_detections(frame_bgr, results)

            # FPS overlay
            fps_text = f"Cam {CAMERA_NUM} | {FRAME_WIDTH}x{FRAME_HEIGHT} | conf>{CONF_THRESH}"
            cv2.putText(frame_bgr, fps_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Weed Detection", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
