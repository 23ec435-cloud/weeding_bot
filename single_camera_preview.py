"""
single_camera_preview.py
Shows live footage from a single Pi Cam 3.
Default: Camera 0. Pass camera index as CLI arg to switch (e.g. python single_camera_preview.py 1)
Press 'q' to quit.
"""

import sys
import cv2
from picamera2 import Picamera2

# --- Config ---
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
# --------------

def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Initialising camera {cam_index}...")

    cam = Picamera2(camera_num=cam_index)
    config = cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    print(f"Camera {cam_index} started. Press 'q' to quit.")

    try:
        while True:
            frame = cam.capture_array()   # shape: (H, W, 3) RGB

            # Convert RGB -> BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.putText(frame_bgr, f"Camera {cam_index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow(f"Pi Cam {cam_index} Preview (press q to quit)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
