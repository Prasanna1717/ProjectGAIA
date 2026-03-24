import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split for pitch metrics")
    parser.add_argument("--model", required=True, help="Path to trained weights, e.g. best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml with test split")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    print("Test metrics:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
