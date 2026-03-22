#!/usr/bin/env bash
set -e

echo "[1/5] Installing system dependencies..."
sudo apt install -y python3-venv python3-pip python3-opencv python3-picamera2 libatlas-base-dev

echo "[2/5] Creating YOLO workspace..."
mkdir -p ~/yolo
cd ~/yolo

echo "[3/5] Creating venv (with system site packages)..."
python3 -m venv --system-site-packages venv

# shellcheck disable=SC1091
source venv/bin/activate

echo "[4/5] Upgrading pip..."
pip install --upgrade pip

echo "[5/5] Installing ultralytics + ncnn..."
pip install ultralytics ncnn

echo "Done. Activate environment with:"
echo "source ~/yolo/venv/bin/activate"
