"""
dual_camera_preview.py
Shows live footage from both Pi Cam 3 cameras side by side.
Camera 0 = left side,  Camera 1 = right side
Press 'q' to quit.
"""

import cv2
import numpy as np
import threading
from picamera2 import Picamera2

# --- Config ---
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
# --------------


class CameraStream:
    """Captures frames in a background thread so both cameras run in parallel."""

    def __init__(self, camera_num, width, height):
        self.cam = Picamera2(camera_num=camera_num)
        self.cam.configure(self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        ))
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
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


def main():
    print("Initialising cameras...")

    stream0 = CameraStream(0, FRAME_WIDTH, FRAME_HEIGHT).start()
    stream1 = CameraStream(1, FRAME_WIDTH, FRAME_HEIGHT).start()

    print("Cameras started. Press 'q' to quit.")

    try:
        while True:
            frame0 = stream0.read()
            frame1 = stream1.read()

            if frame0 is None or frame1 is None:
                continue

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
        stream0.stop()
        stream1.stop()
        cv2.destroyAllWindows()
        print("Cameras stopped.")


if __name__ == "__main__":
    main()
