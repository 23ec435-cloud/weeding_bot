"""
weed_detection_hailo.py
Runs the custom weed detection model on the Hailo-8 (26 TOPS) AI Hat+.

Requires best.hef — compile from best.onnx on an x86-64 Linux machine:
  Step 1 (parse):    hailo parser onnx best.onnx --hw-arch hailo8
  Step 2 (optimize): hailo optimize best.har --hw-arch hailo8 \\
                         --calib-path <folder_with_calibration_images>
  Step 3 (compile):  hailo compiler best.har --hw-arch hailo8
  → Copy the resulting best.hef into this project directory.

If best.hef is not present the script falls back to the generic
yolov8s_h8.hef (COCO model) so you can verify the Hailo pipeline
works before you have the compiled custom model.

Camera:  0 (left) or 1 (right) — set CAMERA_NUM below
Press q to quit.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from picamera2 import Picamera2
from picamera2.devices.hailo import Hailo

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_NUM  = 0          # 0 = left cam, 1 = right cam
FRAME_W     = 1280       # camera capture width  (will be resized for Hailo)
FRAME_H     = 960        # camera capture height
CONF_THRESH = 0.40       # minimum confidence to display a detection

# Custom model class names (must match the order used during training)
CUSTOM_CLASS_NAMES = {0: "Plant", 1: "weeds"}

# Paths
HEF_CUSTOM   = Path(__file__).parent / "best.hef"
HEF_FALLBACK = Path("/usr/share/hailo-models/yolov8s_h8.hef")
# ─────────────────────────────────────────────────────────────────────────────

# BGR draw colours
COLOUR_WEED  = (0,   0,   255)   # red   — spray target
COLOUR_PLANT = (0,   255,   0)   # green — leave alone
COLOUR_OTHER = (255,   0, 255)   # magenta — unknown

WEED_NAMES  = {"weeds", "weed"}
PLANT_NAMES = {"plant", "plants", "crop", "crops"}


def class_colour(name: str) -> tuple:
    n = name.lower()
    if n in WEED_NAMES:
        return COLOUR_WEED
    if n in PLANT_NAMES:
        return COLOUR_PLANT
    return COLOUR_OTHER


def postprocess_hailo_nms(raw: np.ndarray, class_names: dict,
                           input_wh: int, frame_w: int, frame_h: int):
    """
    Decode Hailo YOLOv8 NMS output tensor.

    Tensor shape: (num_classes, 5, max_detections)
      axis 0 — class index
      axis 1 — [y_min, x_min, y_max, x_max, score]  pixel coords in input_wh space
      axis 2 — detection slot (zero-padded when empty)

    Returns list of (x1, y1, x2, y2, score, class_id, class_name)
    scaled to the original frame dimensions.
    """
    sx = frame_w / input_wh
    sy = frame_h / input_wh

    detections = []
    num_classes = raw.shape[0]
    max_dets    = raw.shape[2]

    for cls_id in range(num_classes):
        for det in range(max_dets):
            score = float(raw[cls_id, 4, det])
            if score < CONF_THRESH:
                continue
            y_min = float(raw[cls_id, 0, det])
            x_min = float(raw[cls_id, 1, det])
            y_max = float(raw[cls_id, 2, det])
            x_max = float(raw[cls_id, 3, det])
            # skip zero-padded slots
            if x_max <= 0 and y_max <= 0:
                continue
            name = class_names.get(cls_id, f"class{cls_id}")
            detections.append((
                int(x_min * sx), int(y_min * sy),
                int(x_max * sx), int(y_max * sy),
                score, cls_id, name
            ))
    return detections


def draw_detections(frame: np.ndarray, detections: list):
    """Draw boxes on frame. Returns (annotated_frame, weed_boxes)."""
    weed_boxes = []
    for x1, y1, x2, y2, score, cls_id, name in detections:
        colour  = class_colour(name)
        is_weed = name.lower() in WEED_NAMES
        if is_weed:
            weed_boxes.append((x1, y1, x2, y2, score))

        thickness = 3 if is_weed else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        tag  = "WEED" if is_weed else name
        text = f"{tag} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), colour, -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame, weed_boxes


def main():
    # ── Select HEF ────────────────────────────────────────────────────────────
    if HEF_CUSTOM.exists():
        hef_path    = str(HEF_CUSTOM)
        class_names = CUSTOM_CLASS_NAMES
        print(f"[Hailo] Custom model: {hef_path}")
    else:
        print("=" * 60)
        print("[WARNING] best.hef not found — using COCO fallback model.")
        print("          Detections will NOT be weed/plant-specific!")
        print()
        print("  To compile your custom model (on x86-64 Linux with Hailo DFC):")
        print("    hailo parser onnx best.onnx --hw-arch hailo8")
        print("    hailo optimize best.har --hw-arch hailo8 \\")
        print("        --calib-path <calibration_images_folder>")
        print("    hailo compiler best.har --hw-arch hailo8")
        print("    # → copy best.hef to this project folder")
        print("=" * 60)
        hef_path    = str(HEF_FALLBACK)
        class_names = {i: f"coco_{i}" for i in range(80)}

    # ── Initialise Hailo ──────────────────────────────────────────────────────
    with Hailo(hef_path) as hailo:
        input_shape  = hailo.get_input_shape()   # (H, W, C)
        input_h, input_w = input_shape[0], input_shape[1]
        print(f"[Hailo] Model input: {input_w}×{input_h}")

        inputs, outputs = hailo.describe()
        print(f"[Hailo] Output layers: {[o[0] for o in outputs]}")
        print(f"[Hailo] Output shapes: {[o[1] for o in outputs]}")

        # ── Initialise Camera ─────────────────────────────────────────────────
        cam = Picamera2(camera_num=CAMERA_NUM)
        cam.configure(cam.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
        ))
        cam.start()
        print(f"[Camera] Cam {CAMERA_NUM} started at {FRAME_W}×{FRAME_H} (RGB)")
        print("Running — press q to quit.")

        prev_time = time.time()
        try:
            while True:
                # Capture RGB frame
                frame_rgb = cam.capture_array()

                # Resize to model input size
                resized = cv2.resize(frame_rgb, (input_w, input_h))

                # Hailo inference (blocks until result is ready)
                raw_output = hailo.run(resized)

                # raw_output may be a dict (multiple outputs) or ndarray (single)
                if isinstance(raw_output, dict):
                    raw_output = list(raw_output.values())[0]

                # Decode NMS output
                detections = postprocess_hailo_nms(
                    raw_output, class_names, input_w, FRAME_W, FRAME_H
                )

                # Convert RGB→BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_bgr, weed_boxes = draw_detections(frame_bgr, detections)

                # Log weed detections
                if weed_boxes:
                    print(f"[CAM {CAMERA_NUM}] WEED ×{len(weed_boxes)} → "
                          + ", ".join(f"({x1},{y1},{x2},{y2}) "
                                      f"conf={c:.2f}"
                                      for x1, y1, x2, y2, c in weed_boxes))

                # FPS overlay
                now       = time.time()
                fps       = 1.0 / max(now - prev_time, 1e-9)
                prev_time = now
                overlay   = (f"Cam {CAMERA_NUM} | Hailo-8 26T | "
                             f"{fps:.1f} FPS | "
                             f"Weeds: {len(weed_boxes)}")
                cv2.putText(frame_bgr, overlay, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_bgr, overlay, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                cv2.imshow("Weed Detection — Hailo-8", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cam.stop()
            cv2.destroyAllWindows()
            print("Stopped.")


if __name__ == "__main__":
    main()
