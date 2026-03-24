import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

THIS_DIR = Path(__file__).resolve().parent

from src.gaia_v2.intrusion_engine import Detection, IntrusionEngine
from src.gaia_v2.settings import RuntimeDefaults


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


def parse_resolution(value: str):
    if "x" not in value:
        raise ValueError("Resolution must be in WxH format. Example: 640x480")
    w_text, h_text = value.lower().split("x", 1)
    return int(w_text), int(h_text)


def ensure_alert_dir(root: Path):
    day_dir = root / datetime.now().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def save_snapshot(frame, out_root: Path, class_name: str, conf: float):
    ts = datetime.now().strftime("%H%M%S_%f")
    out_file = out_root / f"{ts}_{class_name}_{conf:.2f}.jpg"
    cv2.imwrite(str(out_file), frame)
    return out_file


def draw_overlay(frame, detections, fps: float, total_events: int):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        color = (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    hud = f"FPS: {fps:.1f} | Detections: {len(detections)} | Events: {total_events}"
    cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 220, 255), 2)
    return frame


def main():
    defaults = RuntimeDefaults()

    parser = argparse.ArgumentParser(description="GAIA V2: Animal/Human intrusion detector for Raspberry Pi")
    parser.add_argument("--model", required=True, help="Path to custom YOLO model (.pt or *_ncnn_model folder)")
    parser.add_argument("--source", default="csi", help="csi, usb0, usb1, 0, or video path")
    parser.add_argument("--resolution", default="640x480", help="Capture resolution, e.g. 640x480")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--conf", type=float, default=defaults.conf_threshold, help="Detection confidence threshold")
    parser.add_argument(
        "--snapshot-conf",
        type=float,
        default=defaults.save_snapshot_conf,
        help="Confidence threshold to save snapshot evidence",
    )
    parser.add_argument("--alerts-dir", default=str(THIS_DIR / "alerts"), help="Directory to store event snapshots")
    parser.add_argument("--cooldown", type=float, default=defaults.event_cooldown_seconds, help="Per-class event cooldown")
    parser.add_argument("--heartbeat", type=int, default=defaults.heartbeat_every_frames, help="Status print interval")
    parser.add_argument("--max-fps", type=float, default=defaults.max_fps, help="Cap processing FPS to reduce Pi CPU load")
    parser.add_argument("--show", action="store_true", help="Show realtime GUI window with boxes")
    args = parser.parse_args()

    os.environ.setdefault("QT_STYLE_OVERRIDE", "Fusion")
    os.environ.setdefault("QT_QPA_PLATFORMTHEME", "")

    width, height = parse_resolution(args.resolution)
    source = parse_source(args.source)

    model = YOLO(args.model, task="detect")
    engine = IntrusionEngine(cooldown_seconds=args.cooldown)

    alerts_root = Path(args.alerts_dir)
    alerts_root.mkdir(parents=True, exist_ok=True)

    cap = None
    picam2 = None
    use_csi = source == "csi"

    if use_csi:
        if Picamera2 is None:
            raise RuntimeError("Picamera2 not available. Install: sudo apt install -y python3-picamera2")
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

    frame_count = 0
    t_prev = time.time()

    try:
        while True:
            if use_csi:
                frame = picam2.capture_array()
                if frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = cap.read()
                if not ok:
                    continue

            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            r = results[0]

            detections = []
            if r.boxes is not None and len(r.boxes) > 0:
                names = r.names if isinstance(r.names, dict) else {}
                xyxy = r.boxes.xyxy.tolist()
                confs = r.boxes.conf.tolist()
                clss = r.boxes.cls.tolist()

                for b, c, cls_idx in zip(xyxy, confs, clss):
                    class_name = names.get(int(cls_idx), str(int(cls_idx)))
                    detections.append(
                        Detection(
                            class_name=class_name,
                            confidence=float(c),
                            bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        )
                    )

            events = engine.evaluate(detections=detections, min_conf=args.conf)
            alert_dir = ensure_alert_dir(alerts_root)

            for ev in events:
                msg = (
                    f"[ALERT] {datetime.fromtimestamp(ev.timestamp).isoformat(timespec='seconds')} "
                    f"class={ev.class_name} conf={ev.confidence:.2f} severity={ev.severity}"
                )
                print(msg)

                if ev.confidence >= args.snapshot_conf:
                    snap = save_snapshot(frame, alert_dir, ev.class_name, ev.confidence)
                    print(f"[SNAPSHOT] saved={snap}")

            frame_count += 1
            now = time.time()
            dt = max(now - t_prev, 1e-6)
            fps = 1.0 / dt
            t_prev = now

            if args.heartbeat > 0 and frame_count % args.heartbeat == 0:
                print(f"[HEARTBEAT] frame={frame_count} fps={fps:.2f} detections={len(detections)} events={len(events)}")

            if args.show:
                vis = frame.copy()
                vis = draw_overlay(vis, detections, fps, len(events))
                cv2.imshow("GAIA Live Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Stopping detector (q pressed)...")
                    break

            if args.max_fps > 0:
                min_frame_time = 1.0 / args.max_fps
                elapsed = time.time() - now
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)

    except KeyboardInterrupt:
        print("Stopping detector...")
    finally:
        if cap is not None:
            cap.release()
        if picam2 is not None:
            picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
