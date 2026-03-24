import argparse
import gc
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import torch
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

# Keep class names exactly equal to folder names.
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


def sanitize_token(text: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def safe_copy(src: Path, dst: Path, max_retries: int = 3):
    """
    Copy file with Windows permission error handling.
    Retries with gc.collect() to release file handles.
    """
    for attempt in range(max_retries):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] PermissionError on copy (attempt {attempt+1}/{max_retries}): {e}. Collecting garbage and retrying...")
                gc.collect()
                time.sleep(0.5)
            else:
                print(f"[ERROR] Failed to copy after {max_retries} attempts: {e}")
                raise


def to_yolo(x1, y1, x2, y2, w, h):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 2.0 or bh < 2.0:
        return None
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def iter_images(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def ensure_dirs(out_root: Path):
    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str, strict_device: bool):
    req = str(requested).strip().lower()
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()

    print("[DEBUG] python=", sys.version.replace("\n", " "))
    print("[DEBUG] torch=", torch.__version__)
    print("[DEBUG] torch.cuda.is_available=", cuda_available)
    print("[DEBUG] torch.cuda.device_count=", cuda_count)
    print("[DEBUG] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[DEBUG] requested_device=", requested)

    if req == "cpu":
        print("[DEBUG] using_device=cpu")
        return "cpu"

    if req in {"0", "1", "2", "3"}:
        if cuda_available and int(req) < cuda_count:
            print(f"[DEBUG] using_device={req}")
            return req
        msg = (
            f"Requested CUDA device '{requested}' but CUDA is not available in this environment. "
            "Install CUDA-enabled PyTorch or use --device cpu."
        )
        if strict_device:
            raise RuntimeError(msg)
        print(f"[WARN] {msg} Falling back to CPU.")
        print("[DEBUG] using_device=cpu")
        return "cpu"

    # Let Ultralytics handle advanced device strings, but still provide visibility.
    print(f"[DEBUG] using_device={requested}")
    return requested


def to_torch_device(device: str) -> str:
    """Convert device string to torch-compatible format."""
    if device == "cpu":
        return "cpu"
    # Convert "0" -> "cuda:0", "1" -> "cuda:1", etc.
    return f"cuda:{device}"


def main():
    parser = argparse.ArgumentParser(description="SOTA off-shelf auto-annotation using YOLO-World on CUDA")
    parser.add_argument("--data-root", required=True, help="Raw class folders root")
    parser.add_argument("--out-root", required=True, help="Output YOLO dataset root")
    parser.add_argument("--world", default="yolov8x-worldv2.pt", help="Primary SOTA model")
    parser.add_argument("--fallback-world", default="yolov8s-worldv2.pt", help="Fallback model")
    parser.add_argument("--device", default="0", help="CUDA device id, e.g. 0")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip files already annotated")
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N images")
    parser.add_argument("--strict-device", action="store_true", help="Fail instead of falling back to CPU when CUDA is unavailable")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    rng = random.Random(args.seed)
    runtime_device = resolve_device(args.device, args.strict_device)

    try:
        world = YOLO(args.world, task="detect")
        print(f"Loaded model: {args.world}")
    except Exception as ex:
        print(f"[WARN] Failed loading {args.world}: {ex}")
        print(f"Loading fallback model: {args.fallback_world}")
        world = YOLO(args.fallback_world, task="detect")

    # Move model to device BEFORE set_classes (which uses CLIP text encoder)
    # Don't use .half() to avoid dtype mismatch during conv+bn fusion in predict()
    if runtime_device != "cpu":
        torch_device = to_torch_device(runtime_device)
        print(f"[DEBUG] Moving model to {torch_device} for text encoder compatibility")
        world.model.to(torch_device)
    else:
        print(f"[DEBUG] Using CPU device")

    total = 0
    with_boxes = 0

    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder = data_root / folder_name
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        imgs = sorted(list(iter_images(folder)))
        if args.max_per_class > 0:
            imgs = imgs[: args.max_per_class]

        world.set_classes(PROMPTS[class_name])
        class_id = class_to_id[class_name]

        class_total = 0
        class_with_boxes = 0

        for idx, img in enumerate(imgs):
            split = "val" if rng.random() < args.val_ratio else "train"
            stem = f"{sanitize_token(class_name)}_{idx:06d}"
            out_img = out_root / split / "images" / f"{stem}{img.suffix.lower()}"
            out_lbl = out_root / split / "labels" / f"{stem}.txt"

            if args.resume and out_img.exists() and out_lbl.exists():
                class_total += 1
                total += 1
                if out_lbl.stat().st_size > 0:
                    class_with_boxes += 1
                    with_boxes += 1
                continue

            safe_copy(img, out_img)
            frame = cv2.imread(str(out_img))
            if frame is None:
                out_lbl.write_text("", encoding="utf-8")
                class_total += 1
                total += 1
                continue

            h, w = frame.shape[:2]
            results = world.predict(
                source=frame,
                conf=args.conf,
                device=runtime_device,
                verbose=False,
            )

            lines = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for b in results[0].boxes.xyxy.tolist():
                    yb = to_yolo(b[0], b[1], b[2], b[3], w, h)
                    if yb is None:
                        continue
                    lines.append(f"{class_id} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")

            out_lbl.write_text("\n".join(lines), encoding="utf-8")

            class_total += 1
            total += 1
            if lines:
                class_with_boxes += 1
                with_boxes += 1

            if args.log_every > 0 and class_total % args.log_every == 0:
                print(
                    f"[PROGRESS] class={class_name} processed={class_total}/{len(imgs)} "
                    f"with_boxes={class_with_boxes}"
                )

        print(
            f"[CLASS DONE] {class_name}: images={class_total} with_boxes={class_with_boxes} "
            f"without_boxes={class_total - class_with_boxes}"
        )

    print("\nAuto-annotation finished.")
    print(f"Total images processed: {total}")
    print(f"Total with boxes: {with_boxes}")
    print(f"Total without boxes: {total - with_boxes}")


if __name__ == "__main__":
    main()
