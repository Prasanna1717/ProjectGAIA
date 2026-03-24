# Wildlife Intrusion Detection - YOLO Training Pipeline

A production-ready machine learning pipeline for training YOLOv8/v11 object detection models on wildlife and human detection. Built for Raspberry Pi deployment with laptop live inference testing.

**Project**: GAIA V2 (Geographic AI-driven Animal Intrusion Alert)  
**Status**: Validated on 10-class wildlife dataset with 82% mAP50 accuracy  
**Framework**: Ultralytics YOLOv11  
**Hardware**: RTX 4060 GPU (training), Raspberry Pi 4B+ (inference)

---

## 📊 Dataset & Results

- **Classes**: 10 animal/human categories (Chital, Dhole, Elephant, Human, Indian Gaur, Leopard, Nilgai, Sloth Bear, Tiger, Wild Boar)
- **Images**: 6,116 annotated images
- **Split**: 70% train (4,281) / 20% val (1,223) / 10% test (612)
- **Test Performance**:
  - mAP50: 0.8198 | mAP50-95: 0.7433
  - Precision: 0.8165 | Recall: 0.7808
- **Inference Speed**: ~5ms per image on RTX 4060

---

## 📁 Repository Structure

```
wildlife-detection-yolo-training/
├── training/                              # All training pipeline scripts
│   ├── annotate_sota_cuda.py             # Zero-shot pseudo-label bootstrap (YOLO-World)
│   ├── bootstrap_autolabel_yoloworld.py  # YOLO-World wrapper for batch labeling
│   ├── create_701020_split.py            # Create train/val/test splits (70/20/10)
│   ├── make_data_yaml.py                 # Generate YAML config for Ultralytics
│   ├── train_export.py                   # Main training + NCNN/ONNX export
│   ├── evaluate_test_set.py              # Test set evaluation & metrics
│   ├── preview_all_classes.py            # Generate QA preview images
│   └── render_label_previews.py          # Visualization utility
│
├── inference/                             # Deployment & inference scripts
│   └── run_rpi_detector.py               # Live camera inference (Pi + laptop GUI)
│
└── README.md                              # This file
```

---

## 🔧 Environment Setup

### Prerequisites
```powershell
# Conda environment with CUDA support
conda create -n gaia-cuda python=3.12 pytorch::pytorch pytorch::pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate gaia-cuda
pip install ultralytics opencv-python pyyaml numpy pillow
```

### Environment File
Create `.env` in your working directory:
```
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=./
```

### GPU Check
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 🚀 Pipeline Walkthrough

### 1️⃣ **Annotation (YOLO-World Bootstrap)**

Start with raw unlabeled images. Generate automatic pseudo-labels using zero-shot YOLO-World:

```powershell
python training/annotate_sota_cuda.py \
  --image_dir "path/to/raw/images" \
  --output_dir "DATASETS/gaia_v2_bootstrap" \
  --device 0 \
  --batch_size 4
```

**Input**: Raw image collection (JPG/PNG)  
**Output**: YOLO format labels (`.txt` files matching each image)  
**Typical Time**: 30-60 min for 6,000 images on RTX 4060

---

### 2️⃣ **Data Splitting (70/20/10)**

Organize into train/val/test splits with stratified sampling (ensures balanced class distribution):

```powershell
python training/create_701020_split.py \
  --input_dir "DATASETS/gaia_v2_bootstrap" \
  --output_dir "DATASETS/gaia_v2_701020" \
  --train_ratio 0.7 \
  --val_ratio 0.2
```

**Input**: Raw images + YOLO labels from Step 1  
**Output**: 
- `train/images/`, `train/labels/`
- `val/images/`, `val/labels/`
- `test/images/`, `test/labels/`
- `data.yaml` (Ultralytics config)

---

### 3️⃣ **QA Preview Generation**

Visualize labeled data to catch annotation errors before training:

```powershell
python training/preview_all_classes.py \
  --data_dir "DATASETS/gaia_v2_701020" \
  --output_dir "DATASETS/gaia_v2_701020/previews" \
  --samples_per_class 10
```

**Output**: Sample images per class with bounding boxes + confidence scores

---

### 4️⃣ **Model Training (YOLO11s)**

Train YOLOv11s with data augmentation and early stopping:

```powershell
$PythonPath = "C:\Users\YourUsername\.conda\envs\gaia-cuda\python.exe"
& $PythonPath "training/train_export.py" `
  --data "DATASETS/gaia_v2_701020/data.yaml" `
  --model "yolov11s.pt" `
  --epochs 120 `
  --imgsz 640 `
  --batch 8 `
  --device 0 `
  --workers 8 `
  --project "runs/gaia_v2" `
  --name "hosur_intrusion_yolo11s" `
  --patience 30
```

**Parameters**:
- `--imgsz`: Input resolution (640, 768, or 1024)
- `--batch`: Batch size (adjust for GPU VRAM: RTX 4060 → max ~12)
- `--patience`: Early stopping epochs (30-40 recommended)
- `--workers`: Data loading threads (8-16 for multicore)

**Output**:
```
runs/gaia_v2/hosur_intrusion_yolo11s/
├── weights/
│   ├── best.pt              # Best checkpoint (by val mAP)
│   ├── best.onnx            # ONNX export (desktop/raspberry pi)
│   └── best_ncnn_model/     # NCNN export (lightweight for Pi)
├── results.csv
└── train_metrics/
```

**Training Time**: ~2-3 hours (120 epochs on RTX 4060)

---

### 5️⃣ **Test Set Evaluation**

Compute final metrics on held-out test split:

```powershell
& $PythonPath "training/evaluate_test_set.py" `
  --model "runs/gaia_v2/hosur_intrusion_yolo11s/weights/best.pt" `
  --data "DATASETS/gaia_v2_701020/data.yaml" `
  --imgsz 640 `
  --device 0
```

**Output**: Per-class mAP50, mAP50-95, precision, recall, confusion matrix

---

### 6️⃣ **Live Inference (Laptop)**

Test real-time detection on laptop/desktop webcam with GUI overlay:

```powershell
& $PythonPath "inference/run_rpi_detector.py" `
  --model "runs/gaia_v2/hosur_intrusion_yolo11s/weights/best.pt" `
  --source 0 `
  --conf 0.5 `
  --show
```

**Arguments**:
- `--source`: `0` (USB webcam) or `1` (integrated camera)
- `--conf`: Confidence threshold (0.0-1.0)
- `--show`: Display live video window (press `q` to quit)
- `--output_dir`: Directory to save alert snapshots

**Output**: Live video overlay with bounding boxes, confidence scores, FPS

---

### 7️⃣ **Raspberry Pi Deployment**

Transfer NCNN model to Raspberry Pi for edge inference:

```bash
# On Raspberry Pi
pip install opencv-python numpy picamera2

python run_rpi_detector.py \
  --model "best_ncnn_model" \
  --source "/dev/video0" \
  --conf 0.5 \
  --output_dir "./alerts"
```

**Model Formats**:
- **NCNN** (fastest on Pi): 36.1 MB
- **ONNX** (cross-platform): 36.3 MB
- **TensorFlow Lite** (optional): Convert from ONNX

---

## 📈 Hyperparameters & Tuning

### Baseline Configuration (640x640)
```yaml
imgsz: 640
batch_size: 8
epochs: 120
patience: 30
augmentation:
  mosaic: 0.7
  mixup: 0.1
  fliplr: 0.5
  degrees: 2
  scale: 0.5
```

### Fine-tuning for Specific Classes (768x768)
```powershell
& $PythonPath "training/train_export.py" `
  --data "DATASETS/gaia_v2_701020/data.yaml" `
  --model "runs/gaia_v2/hosur_intrusion_yolo11s/weights/best.pt" `
  --epochs 50 `
  --imgsz 768 `
  --batch 6 `
  --patience 20
```

### Tips
- **Low VRAM**: Reduce `--batch` or `--imgsz`
- **Higher accuracy**: Increase `--epochs` & `--imgsz`
- **Fast training**: Reduce augmentation intensity (mosaic, mixup)
- **Class imbalance**: Use `--class_weights` or oversample rare classes

---

## 🔍 Troubleshooting

### CUDA Out of Memory
```powershell
# Reduce batch size
--batch 4  # instead of 8
# Or reduce image size
--imgsz 512  # instead of 640
```

### Device Mismatch Error
```powershell
# Always use full device string
--device 0      # ✅ Correct
--device cuda:0 # ✅ Also Correct
--device pytorch.device('cuda:0')  # ❌ Wrong
```

### YOLO-World Text Encoder Issues
YOLO-World requires the text encoder on the same device as the model:
```python
# In annotate_sota_cuda.py
world.model = world.model.to(torch.cuda.current_device())
world.model.set_classes(class_names)
```

### High Training Loss
- Check annotation quality (preview images)
- Verify class distribution (run QA script)
- Increase augmentation (mosaic, mixup)
- Reduce learning rate via `--lr0 0.001`

---

## 📊 Performance Comparison

| Model | imgsz | mAP50 | mAP50-95 | Inference (ms) | NCNN (MB) |
|-------|-------|-------|----------|----------------|-----------|
| YOLO11s (baseline) | 640 | 0.8198 | 0.7433 | 3.7 | 36.1 |
| YOLO11s (fine-tune) | 768 | 0.7658 | 0.6892 | 5.2 | 36.1 |

---

## 📚 Key Scripts Explained

### `train_export.py` (Core Training Engine)
```python
model = YOLO("yolov11s.pt")
results = model.train(
    data=yaml_path,
    epochs=120,
    imgsz=640,
    device=0,
    patience=30,
    augment=True,
    mosaic=0.7,
    mixup=0.1
)
model.export(format='ncnn')  # For Raspberry Pi
model.export(format='onnx')  # Cross-platform
```

### `annotate_sota_cuda.py` (YOLO-World Bootstrap)
Uses zero-shot YOLO-World to generate initial pseudo-labels without manual annotation. Required device management for CLIP text encoder.

### `run_rpi_detector.py` (Live Inference)
Supports both laptop and Raspberry Pi:
- Laptop: USB webcam + OpenCV GUI window
- Pi: CSI camera + optional HDMI display
- Exports alert snapshots on detections

---

## 🎯 Common Workflows

### Quick Start (30 min on GPU)
1. Use pre-trained weights: `yolov11s.pt` (skip Step 1-2)
2. Run fine-training on your data (Step 4)
3. Test on laptop (Step 6)

### Full Pipeline (4-6 hours)
1. Collect raw images
2. Run YOLO-World bootstrap (Step 1)
3. Create splits (Step 2)
4. QA preview (Step 3)
5. Train from scratch (Step 4)
6. Evaluate (Step 5)
7. Deploy to Pi (Step 7)

### Model Selection (45 min)
1. Train baseline at 640x640
2. Train variant at 768x768
3. Evaluate both on test split
4. Pick better model for deployment

---

## 🔐 Production Checklist

Before deploying to Raspberry Pi:
- [ ] Test inference on laptop with various lighting conditions
- [ ] Verify model file exports (NCNN, ONNX)
- [ ] Check confidence thresholds per class
- [ ] Monitor GPU/CPU load during inference
- [ ] Validate alert system (snapshots, logging)
- [ ] Test network connectivity (if sending alerts to server)

---

## 📖 References

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLO-World (Zero-shot Detection)](https://github.com/AILab-CVC/YOLO-World)
- [NCNN Framework (Raspberry Pi)](https://github.com/Tencent/ncnn)
- [Raspberry Pi Object Detection Guide](https://pimylifeup.com/raspberry-pi-object-detection/)

---

## 👤 Author

**Prasanna Vijay** — GAIA V2 Project Lead  
Project GAIA | IEEE EPICS

---

## 📄 License

These scripts are part of the Project GAIA research initiative. For questions or collaborations, please reach out.

---

## 🤝 Contributing

Want to improve the pipeline? Contributions welcome! Areas for enhancement:
- Model compression (quantization, pruning)
- Hardware acceleration (TensorRT, OpenVINO)
- Advanced augmentation strategies
- Real-time alert system integration
- Web dashboard for monitoring

