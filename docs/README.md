# Fall Detection Transformer Project

## Datasets
- **CaucaFall** - Colombian fall detection dataset
- **MCFD** - Multiple Cameras Fall Dataset

## Project Structure
```
fall-detection/
├── config.py              # Configuration
├── utils.py               # Utilities
├── preprocess.py          # Dataset preprocessing
├── preprocess_aio.ipynb   # Jupyter notebook version
├── train.py               # Model training
├── benchmark.py           # Benchmark & evaluation
├── test_model.py          # Model testing
├── app_inference.py       # Real-time inference app
├── check_gpu.py           # GPU check
│
├── src/                   # Source modules
│   ├── pifr_features.py   # PIFR feature extraction
│   ├── hybrid_fall_transformer.py  # Model architecture
│   ├── types.py           # Type definitions
│   └── viz.py             # Visualization utilities
│
├── requirements.txt       # Dependencies
├── README.md              # This file
└── .env                   # Environment variables (Telegram)
```

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Check GPU
python scripts/check_gpu.py

# 2. Preprocess datasets
python preprocess.py
# Or use Jupyter: jupyter notebook notebooks/preprocess_aio.ipynb

# 3. Train model
python train.py

# 4. Benchmark (for thesis)
python benchmark.py

# 5. Test model
python scripts/test_model.py

# 6. Run inference app
python gui_app.py
```

## PIFR Features (60D)
9 geometric angles from 17 COCO keypoints:
- F1, F2: Center of Mass (X, Y)
- F3: Shoulder-Nose Angle
- F4: Torso Angle
- F5: Hip Angle
- F6: Shoulder Angle
- F7, F8: Left/Right Leg Angles
- F9: Nose-to-Ankle Angle

Concatenated with 51 keypoint values = **60D vector**

## Benchmark Metrics
### Accuracy
- Accuracy, Precision, Recall (Sensitivity)
- Specificity, F1-Score
- AUC-ROC, Average Precision
- Confusion Matrix, mAP@0.5

### Efficiency
- Parameters (M), Model Size (MB)
- GFLOPs, FPS, Latency (ms)

## Inference App Features
- Webcam / Video file / RTSP stream
- Real-time keypoint visualization
- Telegram alerting
- Performance profiling
