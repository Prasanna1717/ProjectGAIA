import argparse
import random
import re
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = [
    "CHITAL (Axis axis)",
    "DHOLE (Cuon alpinus)",
    "ELEPHANT (Elephas maximus)",
    "HUMAN (Homo sapiens)",
    "INDIAN GAUR (Bos gaurus)",
    "LEOPARD (Panthera pardus)",
    "NILGAI (Boselaphus tragocamelus)",
    "SLOTH BEAR (Melursus ursinus)",
    "TIGER (Panthera tigris)",
    "WILD BOAR (Sus scrofa)",
]

FOLDER_TO_CLASS = {name: name for name in CLASS_NAMES}

PROMPTS = {
    "CHITAL (Axis axis)": ["chital", "axis deer", "spotted deer"],
    "DHOLE (Cuon alpinus)": ["dhole", "asian wild dog", "wild dog"],
    "ELEPHANT (Elephas maximus)": ["asian elephant", "elephant"],
    "HUMAN (Homo sapiens)": ["person", "human"],
    "INDIAN GAUR (Bos gaurus)": ["gaur", "indian bison", "wild cattle"],
    "LEOPARD (Panthera pardus)": ["leopard", "big cat"],
    "NILGAI (Boselaphus tragocamelus)": ["nilgai", "blue bull", "antelope"],
    "SLOTH BEAR (Melursus ursinus)": ["sloth bear", "bear"],
    "TIGER (Panthera tigris)": ["tiger", "big cat"],
    "WILD BOAR (Sus scrofa)": ["wild boar", "boar", "wild pig"],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def to_yolo(b, w, h):
    x1, y1, x2, y2 = b
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 2 or bh < 2:
        return None
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def iter_images(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def sanitize_token(text: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def main():
    parser = argparse.ArgumentParser(description="Create bootstrap YOLO labels from raw folders using YOLO-World")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--world", default="yolov8x-worldv2.pt")
    parser.add_argument("--fallback-world", default="yolov8s-worldv2.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    try:
        world = YOLO(args.world, task="detect")
        print(f"Loaded world model: {args.world}")
    except Exception as ex:
        print(f"[WARN] failed to load {args.world}: {ex}")
        print(f"Loading fallback world model: {args.fallback_world}")
        world = YOLO(args.fallback_world, task="detect")

    total = 0
    with_boxes = 0

    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder = data_root / folder_name
        if not folder.exists():
            print(f"[WARN] missing folder: {folder}")
            continue

        imgs = sorted(list(iter_images(folder)))
        if args.max_per_class > 0:
            imgs = imgs[: args.max_per_class]

        world.set_classes(PROMPTS[class_name])
        class_id = class_to_id[class_name]

        class_with_boxes = 0
        for idx, img in enumerate(imgs):
            split = "val" if rng.random() < args.val_ratio else "train"
            stem = f"{sanitize_token(class_name)}_{idx:06d}"
            img_out = out_root / split / "images" / f"{stem}{img.suffix.lower()}"
            lbl_out = out_root / split / "labels" / f"{stem}.txt"

            shutil.copy2(img, img_out)
            frame = cv2.imread(str(img_out))
            if frame is None:
                lbl_out.write_text("", encoding="utf-8")
                continue

            h, w = frame.shape[:2]
            res = world.predict(source=frame, conf=args.conf, verbose=False)

            lines = []
            if res[0].boxes is not None and len(res[0].boxes) > 0:
                for b in res[0].boxes.xyxy.tolist():
                    yb = to_yolo(b, w, h)
                    if yb is None:
                        continue
                    lines.append(f"{class_id} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")

            lbl_out.write_text("\n".join(lines), encoding="utf-8")

            total += 1
            if lines:
                class_with_boxes += 1
                with_boxes += 1

        print(f"[CLASS] {class_name}: images={len(imgs)} pseudo_boxes={class_with_boxes}")

    print(f"Done. Images={total}, with_boxes={with_boxes}, without_boxes={total - with_boxes}")
    print("Manual label correction is required before final training.")


if __name__ == "__main__":
    main()
