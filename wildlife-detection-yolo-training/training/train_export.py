import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train GAIA intrusion model and export deployment formats")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/gaia_v2")
    parser.add_argument("--name", default="hosur_intrusion_yolo11s")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--export-openvino-int8", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cos_lr=True,
        mosaic=0.7,
        mixup=0.1,
        degrees=2.0,
        translate=0.05,
        scale=0.5,
        fliplr=0.5,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Missing best weights: {best}")

    best_model = YOLO(str(best))
    ncnn_export = best_model.export(format="ncnn", imgsz=args.imgsz)
    print(f"NCNN export: {ncnn_export}")

    onnx_export = best_model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    print(f"ONNX export: {onnx_export}")

    if args.export_openvino_int8:
        ov_export = best_model.export(format="openvino", int8=True, data=args.data, imgsz=args.imgsz)
        print(f"OpenVINO INT8 export: {ov_export}")

    print("Training + export done.")


if __name__ == "__main__":
    main()
