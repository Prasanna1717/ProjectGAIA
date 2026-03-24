#!/usr/bin/env python3
"""
Generate preview images with bounding boxes for all 10 classes (10 images each).
Output to DATASET_ROOT/previews/ for fast QA across all classes.
"""
import argparse
import random
import re
from pathlib import Path

import cv2

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

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

# Box colors per class
COLORS = [
    (255, 0, 0),    # CHITAL - blue
    (0, 255, 0),    # DHOLE - green
    (0, 0, 255),    # ELEPHANT - red
    (255, 255, 0),  # HUMAN - cyan
    (255, 0, 255),  # INDIAN GAUR - magenta
    (0, 255, 255),  # LEOPARD - yellow
    (128, 0, 255),  # NILGAI - orange
    (255, 128, 0),  # SLOTH BEAR - light blue
    (0, 128, 255),  # TIGER - light green
    (255, 0, 128),  # WILD BOAR - purple
]


def common_name(class_name: str) -> str:
    return class_name.split("(")[0].strip()


def class_stem_prefix(class_name: str) -> str:
    # Must match the naming used in annotation scripts, e.g. INDIAN_GAUR_Bos_gaurus_000123
    return re.sub(r"[^A-Za-z0-9._-]+", "_", class_name).strip("_").lower()


def label_has_class(label_file: Path, class_id: int) -> bool:
    if not label_file.exists() or label_file.stat().st_size == 0:
        return False
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 1:
            continue
        try:
            if int(parts[0]) == class_id:
                return True
        except ValueError:
            continue
    return False


def draw_boxes(image, label_file, img_w, img_h):
    """Draw YOLO format boxes on image."""
    if not label_file.exists():
        return image
    
    lines = label_file.read_text(encoding="utf-8").strip().split("\n")
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            
            # Convert from normalized to pixel coordinates
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            color = COLORS[class_id] if class_id < len(COLORS) else (200, 200, 200)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Show full class name with scientific name, e.g. INDIAN GAUR (Bos gaurus).
            label_text = ID_TO_NAME.get(class_id, "unknown")
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception as e:
            print(f"[WARN] Failed to parse box: {e}")
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate preview images for all classes")
    parser.add_argument("--dataset-root", required=True, help="Root of split dataset (train/val/test)")
    parser.add_argument("--per-class", type=int, default=10, help="Images per class to preview")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    preview_root = dataset_root / "previews"
    preview_root.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    
    # Collect all images across train/val/test
    all_images_by_class = {cid: [] for cid in range(len(CLASS_NAMES))}
    
    for split in ["train", "val", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        
        if not img_dir.exists():
            print(f"[WARN] {split}/images not found")
            continue
        
        for img_file in img_dir.glob("*"):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            
            # Infer class from full sanitized class prefix.
            stem = img_file.stem
            for class_id, class_name in enumerate(CLASS_NAMES):
                if stem.lower().startswith(class_stem_prefix(class_name)):
                    lbl_file = lbl_dir / f"{stem}.txt"
                    has_box = label_has_class(lbl_file, class_id)
                    all_images_by_class[class_id].append((img_file, lbl_file, has_box))
                    break
    
    # Generate previews for each class
    total_rendered = 0
    for class_id, class_name in enumerate(CLASS_NAMES):
        images = all_images_by_class[class_id]
        with_boxes = [x for x in images if x[2]]
        without_boxes = [x for x in images if not x[2]]
        needed = min(args.per_class, len(images))

        sample = []
        if with_boxes:
            sample.extend(rng.sample(with_boxes, min(needed, len(with_boxes))))
        if len(sample) < needed and without_boxes:
            sample.extend(rng.sample(without_boxes, min(needed - len(sample), len(without_boxes))))
        
        class_preview_dir = preview_root / common_name(class_name).replace(" ", "_")
        class_preview_dir.mkdir(exist_ok=True)
        
        print(
            f"[CLASS] {class_name}: generating {len(sample)}/{len(images)} previews "
            f"(with_boxes={len(with_boxes)} without_boxes={len(without_boxes)})"
        )
        
        for idx, (img_file, lbl_file, _) in enumerate(sample):
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"  [SKIP] Failed to read: {img_file.name}")
                continue
            
            h, w = frame.shape[:2]
            frame = draw_boxes(frame, lbl_file, w, h)
            
            out_path = class_preview_dir / f"{idx:03d}_{img_file.name}"
            cv2.imwrite(str(out_path), frame)
            total_rendered += 1
            print(f"  [{idx+1}/{len(sample)}] {out_path.name}")
    
    print(f"\n✅ Preview generation complete: {total_rendered} images in {preview_root}")


if __name__ == "__main__":
    main()
