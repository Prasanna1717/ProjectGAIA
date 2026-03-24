import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Create Ultralytics data.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    out = Path(args.out).resolve()

    lines = [
        f"path: {root.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
