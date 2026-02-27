"""
weed_detection.py
Runs YOLOv8 best.pt model on Pi Cam 3 and displays live detections.

Camera:  0 (left) or 1 (right) — set CAMERA_NUM below
Press 'q' to quit.
"""

import cv2
import numpy as np
import threading
import time
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


class CameraStream:
    """Captures frames in a background thread so inference never stalls the camera."""

    def __init__(self, camera_num, width, height):
        self.cam = Picamera2(camera_num=camera_num)
        self.cam.configure(self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        ))
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.cam.start()
        self._thread.start()
        return self

    def _capture_loop(self):
        while not self._stop.is_set():
            frame = self.cam.capture_array()
            with self._lock:
                self._frame = frame

    def read(self):
        """Return a copy of the most recent frame, or None if not yet available."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cam.stop()


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
    stream = CameraStream(CAMERA_NUM, FRAME_WIDTH, FRAME_HEIGHT).start()
    print("Running — press 'q' to quit.")

    fps_timer   = time.time()
    frame_count = 0
    fps         = 0.0

    try:
        while True:
            frame_rgb = stream.read()
            if frame_rgb is None:
                continue

            # Run YOLOv8 inference (model expects RGB)
            results = model(frame_rgb, verbose=False)

            # Convert to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = draw_detections(frame_bgr, results)

            # Rolling FPS counter
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                fps_timer   = time.time()

            overlay = (f"Cam {CAMERA_NUM} | {FRAME_WIDTH}x{FRAME_HEIGHT} "
                       f"| conf>{CONF_THRESH} | {fps:.1f} FPS")
            cv2.putText(frame_bgr, overlay, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Weed Detection", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
