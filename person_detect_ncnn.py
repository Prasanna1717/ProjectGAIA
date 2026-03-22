import argparse
import os
import time

# Avoid Qt GTK style plugin crashes seen on some Pi/VNC sessions.
os.environ.setdefault("QT_STYLE_OVERRIDE", "Fusion")
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "")

import cv2
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


def parse_resolution(res_text: str):
    if "x" not in res_text:
        raise ValueError("Resolution must be in WxH format, for example 640x480")
    w_text, h_text = res_text.lower().split("x", 1)
    return int(w_text), int(h_text)


def parse_source(source_text: str):
    src = source_text.strip().lower()
    if src == "csi":
        return "csi"
    if src.startswith("usb"):
        idx = src.replace("usb", "")
        return int(idx) if idx else 0
    if src.isdigit():
        return int(src)
    return source_text


def main():
    parser = argparse.ArgumentParser(description="YOLO NCNN person detector for Raspberry Pi")
    parser.add_argument("--model", default="yolo11n_ncnn_model", help="Path to YOLO model or NCNN model dir")
    parser.add_argument("--source", default="csi", help="Camera source (csi, usb0, usb1, 0) or file path")
    parser.add_argument("--resolution", default="640x480", help="Display/capture resolution in WxH")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Disable GUI window (safer on unstable VNC/Qt setups)")
    parser.add_argument("--log-every", type=int, default=30, help="Headless: print status every N frames")
    args = parser.parse_args()

    width, height = parse_resolution(args.resolution)
    source = parse_source(args.source)

    model = YOLO(args.model, task="detect")

    cap = None
    picam2 = None
    use_csi = source == "csi"

    if use_csi:
        if Picamera2 is None:
            raise RuntimeError(
                "Picamera2 is required for --source csi. Install with: sudo apt install -y python3-picamera2"
            )

        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(0.2)
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not cap.isOpened():
            raise RuntimeError(f"Unable to open source: {args.source}")

    prev_time = time.time()
    display_enabled = not args.headless
    warned_display_failure = False
    frame_count = 0

    try:
        while True:
            if use_csi:
                frame = picam2.capture_array()
                if frame is None:
                    print("Failed to grab frame from CSI camera")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to grab frame from source")
                    break

            # class 0 in COCO corresponds to 'person'
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[0],
                verbose=False,
            )

            boxes = results[0].boxes
            person_count = int(len(boxes)) if boxes is not None else 0

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            frame_count += 1

            if not display_enabled:
                if args.log_every > 0 and frame_count % args.log_every == 0:
                    print(f"frame={frame_count} fps={fps:.1f} persons={person_count}")
                continue

            annotated = results[0].plot()

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} persons: {person_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if display_enabled:
                try:
                    cv2.imshow("YOLO Human Detection (NCNN)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except Exception as ex:
                    display_enabled = False
                    if not warned_display_failure:
                        print(f"Display disabled after GUI error: {ex}")
                        print("Continuing in headless mode. Re-run with --headless to avoid GUI calls.")
                        warned_display_failure = True
    finally:
        if cap is not None:
            cap.release()
        if picam2 is not None:
            picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
