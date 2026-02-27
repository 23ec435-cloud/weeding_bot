"""
weed_detection.py
Runs YOLOv8 best.pt model on Pi Cam 3 and displays live detections.

Model classes:
  - weed  → red bounding box  (spray target)
  - plant → green bounding box (leave alone)

Camera:  0 (left) or 1 (right) — set CAMERA_NUM below
Press 'q' to quit.
"""

import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from pathlib import Path

# --- Config ---
CAMERA_NUM    = 0                     # 0 = left cam, 1 = right cam
FRAME_WIDTH   = 416                   # reduced from 640 to lower camera load
FRAME_HEIGHT  = 320                   # reduced from 480
CONF_THRESH   = 0.5                   # minimum confidence to show a detection
MODEL_PATH    = Path(__file__).parent / "best.pt"
TARGET_FPS    = 8                     # cap inference rate (Pi can't sustain full speed)
INFERENCE_SIZE = 320                  # YOLO input size — smaller = much faster on CPU
# --------------

# Class names the model uses for weeds (spray targets) — lowercase for matching
# Model trained with red=weed, green=plant so class names may be "red"/"green"
WEED_CLASS_NAMES  = {"weed", "weeds", "red"}
PLANT_CLASS_NAMES = {"plant", "plants", "crop", "crops", "green"}

# BGR colours
COLOUR_WEED  = (0, 0, 255)    # red   — weed: spray target
COLOUR_PLANT = (0, 255, 0)    # green — plant: leave alone
COLOUR_OTHER = (255, 0, 255)  # magenta — unknown class fallback


def get_class_colour(class_name: str) -> tuple:
    """Return BGR colour based on whether the class is a weed or plant."""
    name = class_name.lower()
    if name in WEED_CLASS_NAMES:
        return COLOUR_WEED
    if name in PLANT_CLASS_NAMES:
        return COLOUR_PLANT
    return COLOUR_OTHER


def is_weed(class_name: str) -> bool:
    return class_name.lower() in WEED_CLASS_NAMES


class CameraCapture:
    """Background thread that continuously captures frames so the main loop
    never blocks waiting for the camera."""

    def __init__(self, camera_num, width, height, fps):
        self.cam = Picamera2(camera_num=camera_num)
        self.cam.configure(self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        ))
        self.cam.start()

        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = False
        self._target_interval = 1.0 / fps

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            frame = self.cam.capture_array()
            with self._lock:
                self._frame = frame
            elapsed = time.time() - t0
            sleep_for = self._target_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def read(self):
        """Return the most recent frame (or None if not ready yet)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)
        self.cam.stop()


def draw_detections(frame, results):
    """Draw bounding boxes and labels onto the frame.

    Returns:
        frame: annotated frame
        weed_boxes: list of (x1, y1, x2, y2, conf) for every weed detected
    """
    weed_boxes = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue

            cls_id = int(box.cls[0])
            label  = result.names[cls_id]
            colour = get_class_colour(label)
            weed   = is_weed(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if weed:
                weed_boxes.append((x1, y1, x2, y2, conf))

            # Bounding box (thicker for weeds to make them stand out)
            thickness = 3 if weed else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

            # Label background
            tag   = "WEED" if weed else label
            text  = f"{tag} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), colour, -1)

            # Label text
            cv2.putText(frame, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame, weed_boxes


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"Classes: {model.names}")
    for idx, name in model.names.items():
        role = "WEED (spray)" if is_weed(name) else ("PLANT (skip)" if name.lower() in PLANT_CLASS_NAMES else "UNKNOWN")
        print(f"  class {idx}: '{name}' → {role}")

    print(f"Starting camera {CAMERA_NUM}...")
    capture = CameraCapture(CAMERA_NUM, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS)
    print("Running — press 'q' to quit.")

    prev_time = time.time()
    try:
        while True:
            loop_start = time.time()

            frame_rgb = capture.read()
            if frame_rgb is None:
                time.sleep(0.05)
                continue                          # camera not ready yet

            # Run YOLOv8 inference at reduced size — much lighter on CPU
            results = model(frame_rgb, imgsz=INFERENCE_SIZE, verbose=False)

            # Convert to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr, weed_boxes = draw_detections(frame_bgr, results)

            # Log weed detections (replace with serial spray command later)
            if weed_boxes:
                print(f"[CAM {CAMERA_NUM}] WEED detected: {len(weed_boxes)} target(s) — "
                      + ", ".join(f"({x1},{y1},{x2},{y2}) conf={c:.2f}"
                                  for x1, y1, x2, y2, c in weed_boxes))

            # Real FPS overlay
            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now
            weed_count = len(weed_boxes)
            fps_text  = (f"Cam {CAMERA_NUM} | {fps:.1f} FPS | conf>{CONF_THRESH} | "
                         f"Weeds: {weed_count}")
            cv2.putText(frame_bgr, fps_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Weed Detection", frame_bgr)

            # Cap the loop to TARGET_FPS — gives CPU breathing room
            elapsed = time.time() - loop_start
            wait_ms = max(1, int((1.0 / TARGET_FPS - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
