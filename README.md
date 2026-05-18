# Elderly Fall Detection using YOLOv11-Pose & Hybrid Transformer

A real-time fall detection system for elderly care, powered by YOLOv11-Pose pose estimation and a temporal Transformer model.

---

## Overview

This project processes video streams to detect falls in real-time using:

1. **YOLOv11-Pose** — Extracts 17-body keypoints per frame
2. **PIFR Features** — Computes 60-dimensional pose representations
3. **Hybrid Transformer** — Models temporal patterns across 60-frame windows
4. **PyQt5 GUI** — Real-time monitoring with Telegram alerts

---

## Project Structure

```
fall-detection/
├── main.py                 # Single entry point (preprocess, train, evaluate, app)
├── requirements.txt        # Project dependencies
├── README.md               # This file
│
├── src/                    # Core source code
│   ├── __init__.py
│   ├── config.py           # Central configuration (paths, hyperparameters)
│   ├── pifr_features.py     # PIFR feature extraction from keypoints
│   ├── hybrid_transformer.py # SOTA Transformer model
│   ├── trainer.py           # Training pipeline
│   ├── evaluator.py        # Benchmark & evaluation
│   ├── gui_app.py          # Real-time PyQt5 GUI
│   └── utils.py            # Utility functions
│
├── data/                   # Data directory (auto-created)
│   ├── raw/
│   │   ├── caucafall/      # CaucaFall dataset
│   │   └── mcfd/           # MCFD dataset
│   └── processed/          # Preprocessed .npy files
│
├── models/                 # Trained model checkpoints
│   └── best_model.pth
│
├── results/                # Evaluation outputs
│   ├── benchmark_comparison.csv
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   └── roc_curve.png
│
├── logs/                   # Training logs
│
└── tests/                  # Unit tests
    └── test_transformer.py
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/fall-detection.git
cd fall-detection

# Install dependencies
pip install -r requirements.txt
```

**Hardware Requirements:**
- Python 3.10+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

---

## Quick Start

### 1. Preprocess Datasets

Convert raw videos into preprocessed `.npy` matrices:

```bash
python main.py --mode preprocess
```

### 2. Train the Model

Train the HybridFallTransformer with online augmentation:

```bash
python main.py --mode train
```

The model uses SOTA hyperparameters:

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `nhead` | 4 |
| `num_layers` | 3 |
| `dropout` | 0.1 |
| `learning_rate` | 5e-4 |
| `weight_decay` | 1e-5 |
| `batch_size` | 64 |
| `early_stopping_patience` | 25 |

### 3. Evaluate

Run full benchmark: accuracy metrics, GFLOPs, FPS, and plots:

```bash
python main.py --mode evaluate
```

### 4. Launch GUI

Start the real-time fall detection application:

```bash
python main.py --mode app
```

**Optional:** Configure Telegram alerts via environment variables:

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

---

## Architecture

```
Video Frame
    │
    ▼
YOLOv11-Pose (17 keypoints)
    │
    ▼
PIFR Feature Extraction (60D)
    │
    ▼
Sliding Window (60 frames, stride=15)
    │
    ▼
Hybrid Transformer (3 layers, 4 heads, d_model=256)
    │
    ▼
Classification: Fall / No-Fall
```

---

## Datasets

- **CaucaFall**: 8,244 videos (2,052 falls, 6,192 non-falls)
- **MCFD (Multiple Cameras Fall Dataset)**: Multi-view fall sequences

---

## Outputs

| File | Description |
|------|-------------|
| `models/best_model.pth` | Trained model weights |
| `results/benchmark_comparison.csv` | Metrics comparison |
| `results/confusion_matrix.png` | Confusion matrix plot |
| `results/pr_curve.png` | Precision-Recall curve |
| `results/roc_curve.png` | ROC curve |

---

## License

MIT License
