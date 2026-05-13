# Hệ thống Phát hiện Ngã - Hybrid YOLOv11-Pose + PIFR + Transformer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv11-green.svg" alt="YOLO">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Chế độ Heuristic (góc)](#chế-độ-heuristic-góc)
  - [Chế độ Transformer (lai ghép)](#chế-độ-transformer-lai-ghép)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [Export ONNX](#export-onnx)
- [Cấu hình Telegram](#cấu-hình-telegram)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

---

## Giới thiệu

Đồ án này trình bày một hệ thống phát hiện ngã người trong thời gian thực sử dụng:

1. **YOLOv11-Pose** - Trích xuất 17 keypoints COCO từ người trong frame
2. **PIFR (Pose-Informed Fall Recognition)** - Trích xuất vector đặc trưng 60 chiều:
   - 51 chiều: 17 keypoints × 3 (x, y, confidence)
   - 9 chiều: Đặc trưng hình học (trọng tâm, góc vai-mũi, góc thân, góc hông, góc vai, góc chân trái/phải, góc mũi-mắt cá)
3. **Transformer** - Phân loại sequence 60 frames để phát hiện ngã
4. **Telegram Alert** - Gửi cảnh báo qua Telegram khi phát hiện ngã

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE 4 GIAI ĐOẠN                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  GIAI ĐOẠN 1 │    │  GIAI ĐOẠN 2 │    │  GIAI ĐOẠN 3 │    │  GIAI ĐOẠN 4 │
│  Preprocess  │───▶│  YOLOv11pose │───▶│  Kinematics  │───▶│   Alerting   │
│  (640x640)   │    │  (17 keypts) │    │  (Angle calc)│    │  (Telegram) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Chi tiết các giai đoạn:

| Giai đoạn | Module | Mô tả |
|-----------|--------|--------|
| 1 | `stage1_preprocess.py` | Resize frame về 640x640 |
| 2 | `stage2_pose.py` | Trích xuất 17 COCO keypoints |
| 3 | `stage3_kinematics.py` | Tính góc, phân loại tư thế |
| 4 | `stage4_alert.py` | Gửi cảnh báo Telegram |

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.9+
- PyTorch 2.0+
- CUDA (khuyến nghị cho training)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/your-repo/fall-detection.git
cd fall-detection

# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Hoặc cài đặt tối thiểu cho CI
pip install -r requirements-ci.txt
```

### Dependencies chính

| Package | Phiên bản | Mục đích |
|---------|-----------|----------|
| ultralytics | ≥8.3.0 | YOLOv11-Pose |
| torch | ≥2.0.0 | Deep learning |
| opencv-python-headless | ≥4.8.0 | Xử lý video |
| PyQt5 | ≥5.15.0 | GUI (app_inference.py) |
| scikit-learn | ≥1.3.0 | Metrics |
| pytest | ≥7.0.0 | Unit tests |

---

## Cấu trúc dự án

```
fall-detection/
├── src/                        # Core modules
│   ├── __init__.py            # Package exports
│   ├── types.py               # Shared dataclasses (FrameDiag)
│   ├── config.py              # PipelineConfig
│   ├── pifr_features.py       # 60-D feature extraction
│   ├── hybrid_fall_transformer.py  # Transformer model
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── stage1_preprocess.py  # Frame preprocessing
│   ├── stage2_pose.py        # YOLOv11 pose extraction
│   ├── stage3_kinematics.py  # Angle computation
│   ├── stage4_alert.py       # Telegram alerting
│   ├── viz.py                # Visualization
│   └── groups.py             # Subject grouping
├── scripts/                    # Utility scripts
│   ├── augmentation.py         # Data augmentation
│   └── export_onnx.py        # ONNX export
├── tests/                      # Unit tests
│   ├── test_pifr_features.py
│   └── test_augmentation.py
├── configs/                    # Configuration files
│   ├── pipeline_config.yaml
│   └── train_config.yaml
├── data/                       # Data directory
│   └── .gitkeep
├── main.py                    # CLI entry point
├── gui_app.py                 # Tkinter GUI
├── app_inference.py           # PyQt5 inference app
├── train_transformer.py        # Training script
├── prepare_dataset.py          # Dataset preparation
├── data_extractor.py          # Feature extraction
├── le2i_zone_based_extractor.py  # Zone-based LE2I extraction
├── evaluate.py                # Comprehensive evaluation
├── benchmark_fps.py           # FPS benchmark on videos
├── final_evaluation.py        # Full eval + SOTA comparison + LaTeX table
├── requirements.txt           # Full dependencies
└── README.md
```

---

## Hướng dẫn sử dụng

### Chế độ Heuristic (góc)

Chạy với Tkinter GUI, sử dụng quy tắc góc + temporal filter, **không cần model Transformer**.

```bash
# Chạy với webcam
python main.py --gui

# Chạy với file video
python main.py --gui --source path/to/video.mp4

# Lật ngang (selfie cam)
python main.py --gui --mirror

# Không hiển thị cửa sổ, chỉ log
python main.py --source 0 --no-show
```

**Ngưỡng mặc định:**
- Góc thân (torso) ≥ 55° → nằm ngang
- Góc mũi-mắt cá ≥ 50° → nằm ngang
- Duy trì tư thế ≥ 60 frames hoặc ≥ 10 giây → xác nhận ngã

### Chế độ Transformer (lai ghép)

Chạy với PyQt5 GUI, sử dụng HybridFallTransformer, **cần `best_hybrid_transformer.pth`**.

```bash
# Chạy app
python main.py --gui-transformer

# Hoặc trực tiếp
python app_inference.py

# Chọn nguồn: webcam, file video, hoặc RTSP stream
```

---

## Chuẩn bị dữ liệu

### AIO Dataset (URFD + GMDCSA-24)

```bash
# Bước 1: Gộp dataset
python prepare_dataset.py \
    --urfd-root data/raw/URFD \
    --gmdcsa-root data/raw/GMDCSA24 \
    --out AIO_Dataset

# Bước 2: Trích đặc trưng
python data_extractor.py \
    --aio-dir AIO_Dataset \
    --out-dir data/processed
```

### LE2I Dataset (Zone-based Protocol)

```bash
# Chuẩn bị LE2I clips
python prepare_le2i_dataset.py \
    --le2i-root data/raw/LE2I \
    --out AIO_Dataset \
    --annotation-csv data/raw/LE2I/LE2I_Fall_Annotation.csv

# Trích đặc trưng với Zone-based Protocol
python le2i_zone_based_extractor.py \
    --aio-dir AIO_Dataset \
    --annotation-json AIO_Dataset/_le2i_annotations.json \
    --out-dir data/le2i_processed \
    --val-subjects 5
```

**Zone-based Protocol:**
| Class | Label | Rule |
|-------|-------|------|
| Fall | 0 | Window bao trùm `[start_fall, end_fall]` |
| Non-Fall (ADL) | 1 | Window kết thúc ≥30 frames trước `start_fall` |
| Discarded | - | Buffer zone, post-fall zone |

---

## Huấn luyện mô hình

```bash
# Huấn luyện cơ bản
python train_transformer.py --data-dir data/processed

# Với data augmentation
python train_transformer.py \
    --data-dir data/processed \
    --augment \
    --aug-temporal-shift-prob 0.5 \
    --aug-noise-prob 0.5 \
    --aug-hflip-prob 0.5

# Với các tham số tùy chỉnh
python train_transformer.py \
    --data-dir data/processed \
    --epochs 200 \
    --batch-size 64 \
    --lr 5e-4 \
    --patience 30 \
    --device cuda
```

**Augmentation parameters (theo Benabdennour et al. 2026):**
- Temporal Shift: ±5 frames (prob=0.5)
- Gaussian Noise: σ=0.01 (prob=0.5)
- Horizontal Flip: left/right keypoint swap (prob=0.5)

**Output:**
- `best_hybrid_transformer.pth` - Model checkpoint
- `best_threshold` - Ngưỡng tối ưu F1

---

## Export ONNX

Export model sang ONNX cho inference trên edge devices (Jetson, RK3568).

```bash
# Export cơ bản
python scripts/export_onnx.py \
    --weights best_hybrid_transformer.pth \
    --out model.onnx

# Với tùy chọn nâng cao
python scripts/export_onnx.py \
    --weights best_hybrid_transformer.pth \
    --out model.onnx \
    --opset 14 \
    --seq-len 60 \
    --feature-dim 60 \
    --skip-validation
```

---

## Cấu hình Telegram

### Cách 1: Biến môi trường

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### Cách 2: File .env

```bash
# Tạo file .env trong thư mục gốc
echo "TELEGRAM_BOT_TOKEN=your_bot_token" > .env
echo "TELEGRAM_CHAT_ID=your_chat_id" >> .env
```

### Cách 3: GUI Settings

Trong giao diện Tkinter, nhấn nút "Settings" và nhập Bot Token và Chat ID.

---

## Chạy tests

```bash
# Cài đặt pytest
pip install pytest scikit-learn

# Chạy tất cả tests
pytest tests/ -v

# Chạy với coverage
pytest tests/ -v --cov=src
```

---

## Dataset References

- **URFD** - UR Fall Detection Dataset
- **GMDCSA-24** - Fall detection dataset (Zenodo)
- **LE2I** - LE2I Fall Detection Dataset

---

## Citation

Nếu sử dụng code này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@thesis{fall_detection_2026,
  title={Hybrid YOLOv11-Pose + PIFR + Transformer Fall Detection},
  author={Fall Detection Team},
  year={2026},
  institution={University}
}
```

---

## Giấy phép

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

<p align="center">
  Made with ❤️ for Fall Detection Research
</p>
