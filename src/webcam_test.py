import time
import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    raise SystemExit(
        "picamera2 is not installed. On Raspberry Pi OS, run:\n"
        "  sudo apt update\n"
        "  sudo apt install -y python3-picamera2\n"
    )

def main():
    picam2 = Picamera2()

    # A good default preview config (fast, compatible with OpenCV)
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)

    picam2.start()
    time.sleep(0.5)  # let auto-exposure settle a bit

    print("Raspberry Pi Camera opened successfully (picamera2/libcamera).")

    frame_count = 0
    fail_count = 0

    try:
        while True:
            frame_count += 1

            try:
                frame = picam2.capture_array()  # RGB array
            except Exception as e:
                fail_count += 1
                print(f"Frame {frame_count} FAILED: {e}")
                continue

            if frame is None or frame.size == 0:
                fail_count += 1
                print(f"Frame {frame_count} FAILED: empty frame")
                continue

            # picamera2 gives RGB; OpenCV expects BGR for correct colors
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if frame_count % 30 == 0:
                print(f"[SUMMARY] Frames: {frame_count}, Failures: {fail_count}")

            cv2.imshow("Pi Camera", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"Program exited successfully. Total frames: {frame_count}, Failed frames: {fail_count}.")

if __name__ == "__main__":
    main()
