import argparse
import random
import re
import shutil
from collections import defaultdict
import math
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def class_from_stem(stem: str):
    # stems look like CLASS_NAME_000123
    return re.sub(r"_\d+$", "", stem)


def find_label(label_dirs, stem: str):
    for d in label_dirs:
        p = d / f"{stem}.txt"
        if p.exists():
            return p
    return None


def ensure_split_dirs(root: Path):
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def largest_remainder_allocation(class_totals, ratio, target_total):
    # Hamilton method: floor each quota, then distribute remainders.
    quotas = {k: v * ratio for k, v in class_totals.items()}
    base = {k: int(math.floor(q)) for k, q in quotas.items()}
    remaining = target_total - sum(base.values())

    remainders = sorted(
        ((k, quotas[k] - base[k]) for k in class_totals),
        key=lambda x: x[1],
        reverse=True,
    )

    i = 0
    while remaining > 0 and i < len(remainders):
        cls = remainders[i][0]
        if base[cls] < class_totals[cls]:
            base[cls] += 1
            remaining -= 1
        i += 1

    return base


def main():
    parser = argparse.ArgumentParser(description="Create deterministic 70/20/10 split from YOLO dataset")
    parser.add_argument("--src-root", required=True, help="Source dataset root with train/ and val/")
    parser.add_argument("--out-root", required=True, help="Output dataset root with train/val/test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--clear-out", action="store_true")
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train+val+test ratios must sum to 1.0")

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    if args.clear_out and out_root.exists():
        shutil.rmtree(out_root)

    ensure_split_dirs(out_root)

    image_dirs = [src_root / "train" / "images", src_root / "val" / "images", src_root / "test" / "images"]
    label_dirs = [src_root / "train" / "labels", src_root / "val" / "labels", src_root / "test" / "labels"]

    images = []
    for d in image_dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                images.append(p)

    if not images:
        raise RuntimeError(f"No images found in {src_root}")

    class_buckets = defaultdict(list)
    for img in images:
        cls = class_from_stem(img.stem)
        class_buckets[cls].append(img)

    rng = random.Random(args.seed)

    class_totals = {cls: len(items) for cls, items in class_buckets.items()}
    total_items = sum(class_totals.values())
    target_train = int(round(total_items * args.train_ratio))
    target_val = int(round(total_items * args.val_ratio))

    train_alloc = largest_remainder_allocation(class_totals, args.train_ratio, target_train)
    val_alloc = largest_remainder_allocation(class_totals, args.val_ratio, target_val)

    summary = {}
    for cls, items in class_buckets.items():
        rng.shuffle(items)
        n = len(items)

        n_train = min(train_alloc.get(cls, 0), n)
        n_val = min(val_alloc.get(cls, 0), n - n_train)
        n_test = n - n_train - n_val

        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        for split, split_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            for img in split_items:
                lbl = find_label(label_dirs, img.stem)
                if lbl is None:
                    continue
                out_img = out_root / split / "images" / img.name
                out_lbl = out_root / split / "labels" / lbl.name
                shutil.copy2(img, out_img)
                shutil.copy2(lbl, out_lbl)

        summary[cls] = {
            "total": n,
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        }

    train_count = len(list((out_root / "train" / "images").glob("*")))
    val_count = len(list((out_root / "val" / "images").glob("*")))
    test_count = len(list((out_root / "test" / "images").glob("*")))

    print(f"Split complete: train={train_count} val={val_count} test={test_count} total={train_count + val_count + test_count}")
    print("CLASS,total,train,val,test")
    for cls in sorted(summary):
        s = summary[cls]
        print(f"{cls},{s['total']},{s['train']},{s['val']},{s['test']}")


if __name__ == "__main__":
    main()
