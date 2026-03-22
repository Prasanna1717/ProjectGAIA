# Raspberry Pi YOLO Human Detection (SSH + VNC)

This setup follows the EJTech guide:
https://www.ejtech.io/learn/yolo-on-raspberry-pi

## What this package includes

- `install_on_pi.sh`: one-time environment and dependency setup.
- `person_detect_ncnn.py`: live person-only detection (class 0) with FPS display.

## Assumptions

- Raspberry Pi OS 64-bit Bookworm.
- You already ran `sudo apt update` and `sudo apt upgrade`.
- You are connected to the Pi via SSH in VS Code.
- VNC server is enabled on Pi (for viewing GUI window remotely).

## 1. Enable camera + VNC + SSH (if not already enabled)

Run on Pi:

```bash
sudo raspi-config
```

Then enable:

- Interface Options -> Camera
- Interface Options -> VNC
- Interface Options -> SSH

Reboot:

```bash
sudo reboot
```

## 2. Copy files to Pi and run setup

If these files are already in your Pi workspace, just run:

```bash
cd "<path-to-CODEBASE AI>"
chmod +x install_on_pi.sh
./install_on_pi.sh
```

## 3. Download YOLO model and export to NCNN

In your Pi terminal:

```bash
cd ~/yolo
source venv/bin/activate

# Download and quick test (creates yolo11n.pt)
yolo detect predict model=yolo11n.pt

# Export for faster Pi inference
yolo export model=yolo11n.pt format=ncnn
```

This creates `yolo11n_ncnn_model/`.

## 4. Run person detection from USB camera

```bash
cd "<path-to-CODEBASE AI>"
source ~/yolo/venv/bin/activate
python3 person_detect_ncnn.py --model ~/yolo/yolo11n_ncnn_model --source usb0 --resolution 640x480 --imgsz 320 --conf 0.4
```

Press `q` in the video window to quit.

## 4b. Run person detection from CSI camera (OV56 on Pi 4B)

For CSI cameras, use `--source csi` (this project script supports Picamera2).

```bash
cd "<path-to-CODEBASE AI>"
source ~/yolo/venv/bin/activate
python3 person_detect_ncnn.py --model ~/yolo/yolo11n_ncnn_model --source csi --resolution 640x480 --imgsz 320 --conf 0.4
```

If Picamera2 is missing:

```bash
sudo apt install -y python3-picamera2
```

## 5. View through VNC

- Open RealVNC Viewer on your laptop.
- Connect to `<pi-ip>`.
- Keep script running via SSH, and the OpenCV window appears on the Pi desktop seen over VNC.

If the app crashes with Qt/GTK errors or segfault in `cv2.waitKey`, run with stable Qt env vars:

```bash
export QT_STYLE_OVERRIDE=Fusion
export QT_QPA_PLATFORMTHEME=
python3 person_detect_ncnn.py --model ~/yolo/yolo11n_ncnn_model --source csi --resolution 640x480 --imgsz 320 --conf 0.4
```

## 5b. Stable GUI view in VNC (browser stream, recommended)

If OpenCV window rendering crashes on your Pi image, use the MJPEG web viewer instead of `cv2.imshow`.

Install Flask once:

```bash
source ~/yolo/venv/bin/activate
pip install flask
```

Run stream server:

```bash
cd "<path-to-CODEBASE AI>"
source ~/yolo/venv/bin/activate
python3 person_detect_web.py --model ~/yolo/yolo11n.pt --source csi --resolution 640x480 --imgsz 320 --conf 0.4 --host 0.0.0.0 --port 5000
```

Open in VNC browser:

```text
http://127.0.0.1:5000
```

Or from your laptop browser (same network):

```text
http://<pi-ip>:5000
```

## Notes

- If camera fails: try `--source usb1` or replug camera.
- For CSI camera issues, run `rpicam-hello -t 5000` to confirm the sensor is detected.
- On older images, the equivalent command is `libcamera-hello -t 5000`.
- If you see `QStandardPaths: wrong permissions on runtime directory /run/user/1000`, fix once with `sudo chmod 700 /run/user/1000`.
- For Pi 4, lower workload: `--resolution 416x320 --imgsz 256`.
- If VNC shows black window, run with a connected display profile or use `vncserver-virtual` setup.

For fully stable operation (no GUI backend usage), run headless:

```bash
python3 person_detect_ncnn.py --model ~/yolo/yolo11n_ncnn_model --source csi --resolution 640x480 --imgsz 320 --conf 0.4 --headless
```

If NCNN still crashes (for example `corrupted double-linked list`), use PyTorch model fallback on this OS stack:

```bash
# If needed, download model file once
yolo detect predict model=yolo11n.pt

# Run same script using .pt model instead of _ncnn_model
python3 person_detect_ncnn.py --model ~/yolo/yolo11n.pt --source csi --resolution 640x480 --imgsz 320 --conf 0.4 --headless
```

The `.pt` fallback is usually slower but more stable than NCNN on some Trixie/Python 3.13 combinations.

## Troubleshooting: "Illegal instruction" when running YOLO

If `yolo detect predict model=yolo11n.pt` exits with `Illegal instruction`, one of the installed Python binary wheels is incompatible with your Pi CPU/OS.

Run these checks first:

```bash
uname -m
cat /etc/os-release | grep VERSION_CODENAME
python3 -V
python3 -c "import platform; print(platform.machine())"
```

Expected architecture is usually `aarch64` on Raspberry Pi OS 64-bit.

Then repair your environment (safe to run as-is):

```bash
cd ~/yolo
source venv/bin/activate

# remove potentially incompatible wheels
pip uninstall -y ultralytics torch torchvision torchaudio ncnn opencv-python opencv-python-headless

# use distro-native scientific stack (more stable on Pi)
# Debian package name is python3-torch (not python3-pytorch)
sudo apt install -y python3-torch python3-torchvision python3-opencv python3-numpy

# install ultralytics without replacing apt torch/opencv
pip install --upgrade pip
pip install ultralytics --no-deps

# optional: install ncnn only after yolo runs successfully
pip install ncnn
```

Quick isolate test (finds which import crashes):

```bash
python3 -c "import torch; print('torch ok', torch.__version__)"
python3 -c "import cv2; print('cv2 ok', cv2.__version__)"
python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
```

If all three commands succeed, retry:

```bash
yolo detect predict model=yolo11n.pt
```

If it still fails, your Pi is likely running a non-64-bit OS or older CPU profile. In that case, reflash Raspberry Pi OS 64-bit Bookworm and rerun this setup.

## Trixie + Python 3.13 note (your current setup)

If your checks show:

- `aarch64`
- `VERSION_CODENAME=trixie`
- `Python 3.13.x`

then `Illegal instruction` is usually caused by an incompatible pip wheel (most often `torch`) in the venv.

Try this exact repair flow:

```bash
cd ~/yolo
source venv/bin/activate

# Remove pip wheels that commonly cause CPU instruction crashes
pip uninstall -y torch torchvision torchaudio ultralytics ncnn opencv-python opencv-python-headless

# Install distro-native builds matched to your OS/CPU
sudo apt update
sudo apt install -y python3-torch python3-torchvision python3-opencv python3-numpy

# Install ultralytics without pulling pip torch/opencv replacements
pip install --upgrade pip
pip install ultralytics --no-deps
```

Validate imports one-by-one:

```bash
python3 -c "import torch; print('torch ok', torch.__version__)"
python3 -c "import cv2; print('cv2 ok', cv2.__version__)"
python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
```

If all imports pass, continue:

```bash
yolo detect predict model=yolo11n.pt
yolo export model=yolo11n.pt format=ncnn
```

If `python3-torch` is unavailable on your Trixie image or import still crashes, first discover available package names:

```bash
apt search python3-torch | head -20
apt search torchvision | head -20
```

Then install the matching names shown by your repository. If torch packages are still unavailable, the fastest stable path is a clean Raspberry Pi OS 64-bit Bookworm install, then rerun this guide.
