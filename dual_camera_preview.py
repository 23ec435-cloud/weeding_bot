"""
dual_camera_preview.py
Shows live footage from both Pi Cam 3 cameras side by side.
Camera 0 = left side,  Camera 1 = right side
Press 'q' to quit.
"""

import cv2
import numpy as np
from picamera2 import Picamera2

# --- Config ---
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
# --------------

def main():
    print("Initialising cameras...")

    cam0 = Picamera2(camera_num=0)
    cam1 = Picamera2(camera_num=1)

    config0 = cam0.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    config1 = cam1.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )

    cam0.configure(config0)
    cam1.configure(config1)

    cam0.start()
    cam1.start()
    print("Cameras started. Press 'q' to quit.")

    try:
        while True:
            frame0 = cam0.capture_array()   # shape: (H, W, 3) RGB
            frame1 = cam1.capture_array()

            # Convert RGB -> BGR for OpenCV display
            frame0_bgr = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
            frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

            # Add labels
            cv2.putText(frame0_bgr, "Camera 0 - Left",  (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame1_bgr, "Camera 1 - Right", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Stack side by side
            combined = np.hstack((frame0_bgr, frame1_bgr))

            cv2.imshow("Dual Pi Cam Preview (press q to quit)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam0.stop()
        cam1.stop()
        cv2.destroyAllWindows()
        print("Cameras stopped.")

if __name__ == "__main__":
    main()
