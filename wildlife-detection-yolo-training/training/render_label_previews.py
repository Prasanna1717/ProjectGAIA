import argparse
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

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def draw_yolo_boxes(img, labels):
    h, w = img.shape[:2]
    for line in labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        x, y, bw, bh = map(float, parts[1:])

        x1 = int((x - bw / 2.0) * w)
        y1 = int((y - bh / 2.0) * h)
        x2 = int((x + bw / 2.0) * w)
        y2 = int((y + bh / 2.0) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def process_split(dataset_root: Path, split: str, out_dir: Path, max_samples: int):
    labels_dir = dataset_root / split / "labels"
    images_dir = dataset_root / split / "images"

    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    count = 0
    for lbl in labels_dir.glob("*.txt"):
        if lbl.stat().st_size <= 0:
            continue
        lines = [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue

        img_path = find_image(images_dir, lbl.stem)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = draw_yolo_boxes(img, lines)
        out_path = out_split / f"{lbl.stem}_preview.jpg"
        cv2.imwrite(str(out_path), img)

        count += 1
        if count >= max_samples:
            break

    return count


def main():
    parser = argparse.ArgumentParser(description="Render YOLO label previews with bounding boxes")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples-per-split", type=int, default=15)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = process_split(dataset_root, "train", out_dir, args.samples_per_split)
    n_val = process_split(dataset_root, "val", out_dir, args.samples_per_split)

    print(f"Rendered previews: train={n_train}, val={n_val}, total={n_train + n_val}")


if __name__ == "__main__":
    main()
