import argparse
import time

import cv2
from flask import Flask, Response
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


def parse_resolution(res_text: str):
    if "x" not in res_text:
        raise ValueError("Resolution must be WxH, e.g. 640x480")
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


class DetectorStream:
    def __init__(self, model_path: str, source, width: int, height: int, imgsz: int, conf: float):
        self.model = YOLO(model_path, task="detect")
        self.source = source
        self.width = width
        self.height = height
        self.imgsz = imgsz
        self.conf = conf

        self.cap = None
        self.picam2 = None
        self.use_csi = source == "csi"

        if self.use_csi:
            if Picamera2 is None:
                raise RuntimeError("Picamera2 required for CSI source. Install: sudo apt install -y python3-picamera2")
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.2)
        else:
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Unable to open source: {source}")

        self.prev_time = time.time()

    def read_frame(self):
        if self.use_csi:
            frame = self.picam2.capture_array()
            if frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ok, frame = self.cap.read()
        return frame if ok else None

    def infer_annotate(self, frame):
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[0],
            verbose=False,
        )
        boxes = results[0].boxes
        person_count = int(len(boxes)) if boxes is not None else 0

        annotated = results[0].plot()
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

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
        return annotated

    def mjpeg_generator(self):
        while True:
            frame = self.read_frame()
            if frame is None:
                continue

            annotated = self.infer_annotate(frame)
            ok, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            chunk = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
            )

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()


def main():
    parser = argparse.ArgumentParser(description="YOLO person detection MJPEG stream for Raspberry Pi")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to model (.pt or NCNN model folder)")
    parser.add_argument("--source", default="csi", help="csi, usb0, usb1, 0, or video path")
    parser.add_argument("--resolution", default="640x480", help="Capture resolution WxH")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5000, help="Bind port")
    args = parser.parse_args()

    width, height = parse_resolution(args.resolution)
    source = parse_source(args.source)

    stream = DetectorStream(args.model, source, width, height, args.imgsz, args.conf)
    app = Flask(__name__)

    @app.route("/")
    def index():
        return (
            "<html><head><title>YOLO Stream</title></head>"
            "<body style='margin:0;background:#111;color:#eee;font-family:Arial;'>"
            "<div style='padding:8px 12px;'>YOLO Human Detection Stream</div>"
            "<img src='/video' style='width:100%;max-width:1280px;display:block;margin:auto;'/>"
            "</body></html>"
        )

    @app.route("/video")
    def video():
        return Response(stream.mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        stream.close()


if __name__ == "__main__":
    main()