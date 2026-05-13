# Project Context: Hybrid YOLOv11-Pose + PIFR + Transformer Fall Detection System

> **Version**: May 2026
> **Project Repository**: `fall-detection`
> **Git Working Directory**: `E:\fall-detection` (local) / `E:/fall-detection` (cross-platform)
> **Kaggle Working**: `/kaggle/working`
> **Datasets**: URFD, GMDCSA-24, LE2I (via Kaggle Datasets)

---

## 1. System Architecture

### 1.1 Overview: 2-Stage Pipeline

The system is a **hybrid computer vision + deep learning pipeline** for real-time human fall detection in video sequences. It operates in two stages:

| Stage | Module | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| **Stage 1** | YOLOv11-Pose + PIFR Feature Extraction | Video frames (BGR) | 60-D feature vector per frame | Per-frame keypoint extraction and geometric engineering |
| **Stage 2** | HybridFallTransformer Classification | Sequence of 60 frame vectors (60, 60) | Binary fall probability | Temporal sequence modeling via Transformer |

**End-to-end pipeline flow:**

```
Video Frame (BGR)
    │
    ▼
┌─────────────────────────┐
│ Stage 1: Preprocessing  │  Resize to 640×640 (IMGSZ=640)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ YOLOv11-Pose (17 KPIs)  │  COCO 17-keypoint extraction
│                         │  Normalize x,y to [0,1] relative to frame
│                         │  MIN_MEAN_CONF=0.2 filter
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ PIFR 60-D Vector        │  51 keypoint dims + 9 geometric dims
│                         │  GeometricFeatureExtractor class
└─────────────────────────┘
    │
    ▼  Sliding Window: T=60, stride=15
┌─────────────────────────┐
│ HybridFallTransformer   │  TransformerEncoder × 3 layers
│ (B, 60, 60) → (B, 1)  │  BCEWithLogitsLoss → sigmoid
└─────────────────────────┘
    │
    ▼
Fall Alert (Telegram) / No-Fall
```

### 1.2 Stage 1: YOLOv11-Pose Keypoint Extraction

**Model**: `yolo11n-pose.pt` (Ultralytics YOLOv11n-Pose, nano variant)
- **Parameters**: ~2.8M params (Pose)
- **GFLOPs**: ~3.2 (Pose branch)
- **Model size**: ~6.26 MB

**COCO 17 Keypoints extracted per frame:**

| Index | Name | Description |
|-------|------|-------------|
| 0 | NOSE | Nose |
| 1 | L_EYE | Left eye |
| 2 | R_EYE | Right eye |
| 3 | L_EAR | Left ear |
| 4 | R_EAR | Right ear |
| 5 | L_SHOULDER | Left shoulder |
| 6 | R_SHOULDER | Right shoulder |
| 7 | L_ELBOW | Left elbow |
| 8 | R_ELBOW | Right elbow |
| 9 | L_WRIST | Left wrist |
| 10 | R_WRIST | Right wrist |
| 11 | L_HIP | Left hip |
| 12 | R_HIP | Right hip |
| 13 | L_KNEE | Left knee |
| 14 | R_KNEE | Right knee |
| 15 | L_ANKLE | Left ankle |
| 16 | R_ANKLE | Right ankle |

**Normalization**: Coordinates (x, y) are divided by frame width/height respectively → range [0, 1].

**Quality filtering**: Frames with mean keypoint confidence < 0.2 are imputed with the previous valid frame's vector (within-clip propagation only; no inter-clip propagation).

### 1.3 PIFR: 60-Dimensional Pose-Informed Feature Engineering

**Pose-Informed Fall Recognition (PIFR)** produces a 60-D vector per frame combining raw keypoint data with hand-crafted geometric features.

**Vector composition:**

| Range | Dimensions | Content |
|-------|------------|---------|
| `[0:51]` | 51 | 17 keypoints × 3 (x, y, confidence) — flattened row-major |
| `[51:60]` | 9 | Geometric features |

**9 Geometric Features:**

| Index | Name | Definition | Formula |
|-------|------|-----------|---------|
| 51 | `center_mass_x` | Center of mass X of valid keypoints | mean(x_i) |
| 52 | `center_mass_y` | Center of mass Y of valid keypoints | mean(y_i) |
| 53 | `shoulder_nose_angle` | Angle at nose between shoulders | arccos((BA·BC)/(\|BA\|×\|BC\|)) |
| 54 | `torso_angle` | Torso tilt vs. vertical axis | arccos(v_y / \|v\|), v = mid_hip − nose |
| 55 | `hip_angle` | Hip line angle vs. horizontal | arccos(v_x / \|v\|), v = R_hip − L_hip |
| 56 | `shoulder_angle` | Shoulder line angle vs. horizontal | arccos(v_x / \|v\|), v = R_shoulder − L_shoulder |
| 57 | `left_leg_angle` | Angle at left knee | arccos((L_knee−L_hip)·(L_ankle−L_knee) / (products)) |
| 58 | `right_leg_angle` | Angle at right knee | same formula for right side |
| 59 | `nose_to_ankle_angle` | Body axis tilt vs. vertical | arccos(v_y / \|v\|), v = mid_ankle − nose |

All geometric angle features are normalized to `[0, 1]` by dividing by π (180°). This Min-Max normalization handles varying body proportions across subjects and video resolutions.

**Key property**: The 9 geometric features encode **invariant relational information** (angles, ratios) rather than absolute positions, making them robust to camera placement and person size variation.

### 1.4 Stage 2: HybridFallTransformer Classification

**Input shape**: `(B, 60, 60)` — batch of B sequences, each 60 frames × 60 features.

**Architecture:**

```
Input (B, 60, 60)
    │
    ├─ Linear(60 → 256)             # Input projection
    ├─ × √256                        # Scale embedding by d_model^0.5
    ├─ + SinusoidalPositionalEncoding(256, max_len=60)
    │       PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    │       PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    │
    ├─ TransformerEncoderLayer × 3
    │       d_model=256, nhead=4, dim_ff=256, dropout=0.1, activation=relu
    │
    ├─ Mean Pooling over time (dim=1)  # Shape: (B, 256)
    │
    └─ MLP Head
          Linear(256 → 32) → ReLU → Dropout(0.1) → Linear(32 → 1)
    │
    ▼
Output (B, 1) — binary logit (raw, unactivated)
```

**Why Transformer?** Unlike LSTMs that process sequences sequentially, the Transformer uses self-attention to model global temporal dependencies in a single forward pass. This captures long-range fall dynamics (standing → losing balance → impact → lying) more efficiently than recurrent models.

**Key design choices:**
- **Sinusoidal PE** (Vaswani et al., 2017): Fixed positional encodings that generalize to unseen sequence lengths, unlike learned embeddings.
- **Mean pooling**: Aggregates the entire 60-frame sequence into a single representation, suitable for binary classification.
- **3 Transformer layers**: Enough to model fall dynamics without excessive computational cost.

### 1.5 Dual Inference Modes

The system supports two real-time inference modes:

**Mode 1: Rule-based (Tkinter GUI) — No training required**
- Uses geometric angle thresholds without any model
- Torso angle threshold: 55° from vertical
- Nose-to-ankle angle threshold: 50° from vertical
- Temporal filter: minimum 60 frames or 10 seconds in "laydown" posture
- Source: `main.py`, `gui_app.py`

**Mode 2: Transformer (PyQt5 GUI) — Requires `best_hybrid_transformer.pth`**
- Sliding window: T=60, stride=15
- Inference threshold: 0.18 (alert) / optimal tuned threshold from checkpoint
- Hybrid heuristic filtering (posture + shape analysis) to suppress false alarms
- Telegram async alerting with cooldown
- Source: `app_inference.py`

---

## 2. Dataset & Preprocessing

### 2.1 Supported Datasets

| Dataset | Full Name | Format | Source |
|---------|-----------|--------|--------|
| **URFD** | UR Fall Detection Dataset | Zipped frame folders | `/kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD/` |
| **GMDCSA-24** | GMDCSA Fall Dataset | `.mp4` videos | `/kaggle/input/datasets/phmthduyn/fall-detection-dataset/GMDCSA24/` |
| **LE2I** | LE2I Fall Detection Dataset | `.avi` videos with frame annotations | `/kaggle/input/datasets/tuyenldvn/falldataset-imvia/` |

**Local paths** (when not on Kaggle):
- `FALL_DATASET_ROOT` = `/kaggle/input/datasets/phmthduyn/fall-detection-dataset` (URFD + GMDCSA)
- `LE2I_DATASET_ROOT` = `/kaggle/input/datasets/tuyenldvn/falldataset-imvia`

### 2.2 AIO (All-in-One) Dataset Fusion

All datasets are converted to a unified **AIO_Dataset** structure:

```
AIO_Dataset/
├── fall/           # label=1
│   ├── urfd_fall_*/         # from URFD/Fall/
│   ├── gmdcsa_subjectN_*/    # from GMDCSA24/Subject N/Fall/
│   └── le2i_scene_*/        # from LE2I fall videos
├── nofall/         # label=0
│   ├── urfd_adl_*/           # from URFD/ADL/
│   ├── gmdcsa_subjectN_*/    # from GMDCSA24/Subject N/ADL/
│   └── le2i_scene_*/         # from LE2I nofall videos
└── _le2i_annotations.json    # LE2I metadata (start_fall, end_fall)
```

**Output naming convention**:
- URFD: `urfd_fall_*`, `urfd_adl_*`
- GMDCSA: `gmdcsa_subject{N}_*`
- LE2I: `le2i_{scene_name}_*`

### 2.3 Keypoint Normalization Strategy

```
frame_bgr → resize(640, 640) → YOLOv11-Pose → (17, 3) keypoints [x, y, conf]
→ k[:, 0] /= frame_width   → x ∈ [0, 1]
→ k[:, 1] /= frame_height  → y ∈ [0, 1]
→ Geometric angles computed → normalized by π → ∈ [0, 1]
```

**Low-confidence frame handling**: Frame-level imputation within the same clip only. The previous valid frame's 60-D vector is used. Frames at the start of a clip with no prior valid frame are discarded from that clip.

### 2.4 Zone-Based Temporal Protocol (IEEE 2026)

For **LE2I** dataset (which has frame-level fall annotations), a strict **Zone-Based Protocol** is used to generate high-quality training sequences with unambiguous labels.

**Motivation**: Standard sliding windows create ambiguous labels at fall boundaries. A window partially overlapping with the fall phase could be labeled either way, introducing noise.

**Protocol definition** (per video with `start_fall` and `end_fall` frame indices):

```
Timeline:  [0]--------[start_fall-30]----[start_fall]---[end_fall]---[total_frames]
           |         Zone A         Zone B   Zone C    Zone D         |
           |<----- ADL Zone ----><-Buffer-><-Fall Zone-><-Post-Fall->|
           |    Label=1 (NoFall)  DISCARD  Label=0 (Fall)  DISCARD   |
```

| Zone | Condition | Label | Action |
|------|-----------|-------|--------|
| **ADL Zone** | `window_end ≤ start_fall − 30` | Class 1 (Non-Fall) | Include |
| **Buffer Zone** | `start_fall − 30 < window_end < start_fall` | — | **DISCARD** |
| **Fall Zone** | `window_start ≤ start_fall AND window_end ≥ end_fall` | Class 0 (Fall) | Include |
| **Post-Fall Zone** | `window_start > end_fall` | — | **DISCARD** |

**Sliding window parameters**:
- `SEQ_LEN = 60` — fixed window length (frames)
- `STRIDE = 15` — sliding stride (frames)
- `FALL_SAFETY_MARGIN = 30` — 30-frame buffer before `start_fall` for ADL class

**Implementation** (`le2i_zone_based_extractor.py`):
- Reads annotation JSON (`_le2i_annotations.json`) per video
- For each video, iterates windows: `start=0, stride=15`
- Classifies each window by zone membership
- Rejects windows in Buffer/Post-Fall zones
- Resamples window to exactly 60 frames using linear interpolation (`resample_to_length()`)

**Group-based train/val split**: To prevent **subject leakage**, train/val split is done at the subject/scene level using `GroupShuffleSplit`:
- URFD: each clip = its own group
- GMDCSA: `subject{N}` = same group
- LE2I: `scene_name` = same group

---

## 3. Hyperparameters

### 3.1 Model Architecture Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Transformer embedding dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers` | 3 | Number of Transformer encoder layers |
| `dim_feedforward` | 256 | Feed-forward hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `activation` | ReLU | Feed-forward activation |
| `seq_len` | 60 | Sliding window length (frames) |
| `feature_dim` | 60 | PIFR feature dimension |
| `in_features` | 60 | Model input projection |
| `max_len` | 60 | Sinusoidal PE maximum length |

### 3.2 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | **AdamW** | Per Benabdennour et al. (2026) |
| Learning rate | **5e-4** (0.0005) | `train_transformer.py --lr` |
| Weight decay | **1e-5** (0.00001) | `train_transformer.py --weight-decay` |
| Loss function | **BCEWithLogitsLoss** | Per Benabdennour et al. (2026) |
| Epochs | **100** (max) | `train_transformer.py --epochs` |
| Early stopping patience | **25** | `train_transformer.py --patience` |
| Batch size | **64** | `train_transformer.py --batch-size` |
| Validation ratio | **0.2** | `train_transformer.py --val-ratio` |
| Random seed | **42** | `train_transformer.py --seed` |
| num_workers | 4 | DataLoader workers |
| Device | auto (CUDA/CPU) | `train_transformer.py --device` |

### 3.3 Data Augmentation Parameters

Applied **only during training** (validation set uses original data).

| Augmentation | Parameter | Default | Description |
|-------------|-----------|---------|-------------|
| Temporal Shift | `aug_temporal_shift_prob` | 0.5 | Random roll ±5 frames along time axis |
| Temporal Shift max | `aug_temporal_shift_max` | 5 | Max frames to shift (±5) |
| Gaussian Noise | `aug_noise_prob` | 0.5 | Camera jitter simulation |
| Gaussian σ | `aug_noise_sigma` | **0.01** | Per PLOS ONE 2026 recommendation |
| Horizontal Flip | `aug_hflip_prob` | 0.5 | Flip X coords + swap left/right keypoint pairs |

**Horizontal flip implementation**:
- X coordinates: `new_X = 1.0 - X`
- Left/right keypoint pairs swapped: (L_EYE↔R_EYE), (L_EAR↔R_EAR), (L_SHOULDER↔R_SHOULDER), etc.
- `left_leg_angle` (index 57) ↔ `right_leg_angle` (index 58) swapped
- Center-based features (torso_angle, hip_angle, etc.) remain unchanged

### 3.4 Inference Runtime Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `INFER_STRIDE` | 15 | Sliding window stride during inference |
| `MIN_VALID_FRAMES_FOR_INFER` | 8 | Min frames before first inference |
| `MAX_MISSING_FRAMES` | 15 | Frames before buffer clear |
| `ALERT_COOLDOWN_SEC` | 10.0 | Telegram cooldown between alerts |
| `ALERT_MIN_PROB` | 0.18 | Minimum prob to trigger alert |

---

## 4. Actual Evaluation Results

### 4.1 Checkpoint Metadata

| Field | Value |
|-------|-------|
| Checkpoint path | `best_hybrid_transformer.pth` |
| Training samples | **160** |
| Feature shape | **(160, 60, 60)** |
| Label shape | **(160, 1)** |
| Groups | **160** (subject-level, preventing leakage) |
| Tuned threshold | **0.2300** |
| Best val F1 | **0.9388** |
| Best val F1 (tuned) | **0.9388** |

### 4.2 Validation Metrics (from `metrics.json`)

| Metric | Value | Formula |
|--------|-------|---------|
| **F1-Score** | **0.9241** | 2×(P×R)/(P+R) |
| **Precision** | **0.9241** | TP/(TP+FP) |
| **Recall** | **0.9241** | TP/(TP+FN) |
| **ROC-AUC** | **0.9526** | Area under ROC curve |
| **PR-AUC** | **0.9412** | Average precision score |
| **Optimal Threshold** | **0.2300** | Tuned on validation set |

### 4.3 Confusion Matrix

|  | Predicted: No-Fall | Predicted: Fall |
|--|---------------------|------------------|
| **Actual: No-Fall** | TN = 75 | FP = 6 |
| **Actual: Fall** | FN = 6 | TP = 73 |

- **Total samples**: 160
- **TN (True Negatives)**: 75
- **FP (False Positives)**: 6
- **FN (False Negatives)**: 6
- **TP (True Positives)**: 73

### 4.4 Model Complexity

| Metric | Value |
|--------|-------|
| Transformer input | (B, 60, 60) |
| Transformer embedding | d_model = 256 |
| Attention heads | 4 |
| Transformer layers | 3 |
| Total Transformer params | ~1.5M (estimated from architecture) |
| Pose model params | ~2.8M (YOLOv11n-Pose) |
| Pose model size | ~6.26 MB |
| Transformer model size | ~6 MB (`.pth`) |

### 4.5 Inference Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **ONNX Opset** | 14 | Modern ONNX operators |
| **ONNX dynamic axes** | Batch size | Dynamic batch, static seq_len/feature_dim |
| **Frame throughput** | Video FPS | Dependent on YOLO + Transformer latency |
| **Alert latency** | ~30 ms | Telegram async send |

---

## 5. Edge Deployment & Alerts

### 5.1 ONNX Export

**Script**: `scripts/export_onnx.py`

```bash
python scripts/export_onnx.py \
    --weights best_hybrid_transformer.pth \
    --out hybrid_transformer.onnx \
    --opset 14 \
    --seq-len 60 \
    --feature-dim 60
```

**Export configuration:**
- Input shape: `(1, 60, 60)` — dynamic batch, static sequence length
- Output shape: `(1, 1)` — binary logit
- Opset version: 14 (supports modern ONNX operators)
- `do_constant_folding=True` — fold constants for optimization
- `export_params=True` — include model weights

**Validation**: Exports with PyTorch → ONNX comparison (`validate_pytorch_vs_onnx()`) using ONNX Runtime with optimized providers (CUDAExecutionProvider > CPUExecutionProvider).

### 5.2 Telegram Bot Integration

**Setup** (two methods):
1. **Environment variables**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
2. **`.env` file** in project root:

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Alert flow** (`app_inference.py`):
1. Model outputs `prob ≥ alert_prob_threshold` AND
2. Hybrid heuristic confirms fall-like posture AND
3. NOT sofa-like (sitting) → suppress false alarm
4. `TelegramNotifier.maybe_notify_async()` called with:
   - Snapshot frame with keypoint overlay + red box
   - Caption: `Fall p={prob:.3f}`
   - Cooldown: 10 seconds between sends

**Async design**: Alerts sent in background thread (`ThreadPoolExecutor`) to avoid blocking inference.

### 5.3 Dual Deployment Modes

| Feature | Tkinter GUI (Rule-based) | PyQt5 GUI (Transformer) |
|---------|--------------------------|-------------------------|
| Model required | No | Yes (`best_hybrid_transformer.pth`) |
| Detection method | Geometric thresholds | Transformer + heuristics |
| Alert system | Basic | Telegram + hybrid filtering |
| Video source | Webcam / file | Webcam / file / RTSP stream |
| Entry point | `main.py --gui` | `app_inference.py` |

---

## 6. Complete Training Pipeline

### 6.1 Kaggle 8-Step Pipeline (`kaggle_train.py`)

| Step | Script | Description |
|------|--------|-------------|
| 1 | — | Dataset structure validation |
| 2 | `prepare_dataset.py` | URFD + GMDCSA → AIO_Dataset |
| 3 | `prepare_le2i_dataset.py` | LE2I → AIO_Dataset + `_le2i_annotations.json` |
| 4 | `data_extractor.py` | YOLO keypoint extraction → `data/processed/` |
| 5 | `le2i_zone_based_extractor.py` | Zone-based sliding windows → `data/le2i_processed/` |
| 6 | — | Merge AIO + LE2I features → `data/merged/` |
| 7 | `train_transformer.py` | Train HybridFallTransformer → checkpoint |
| 8 | `scripts/export_onnx.py` | Export ONNX (optional) |

### 6.2 Key Constants Summary

| Constant | Value | Location |
|----------|-------|----------|
| `IMGSZ` | 640 | `src/pifr_features.py` |
| `MIN_MEAN_CONF` | 0.2 | `src/pifr_features.py` |
| `FEATURE_DIM` | 60 | `src/pifr_features.py` |
| `SEQ_LEN` | 60 | `src/pifr_features.py`, `src/hybrid_fall_transformer.py` |
| `EPS` | 1e-6 | `src/pifr_features.py` |
| `d_model` | 256 | `src/hybrid_fall_transformer.py` |
| `nhead` | 4 | `src/hybrid_fall_transformer.py` |
| `num_layers` | 3 | `src/hybrid_fall_transformer.py` |
| `dim_feedforward` | 256 | `src/hybrid_fall_transformer.py` |
| `dropout` | 0.1 | `src/hybrid_fall_transformer.py` |
| `STRIDE` | 15 | `le2i_zone_based_extractor.py` |
| `FALL_SAFETY_MARGIN` | 30 | `le2i_zone_based_extractor.py` |
| `ALERT_COOLDOWN_SEC` | 10.0 | `app_inference.py` |

---

## 7. File Structure Reference

```
fall-detection/
├── src/
│   ├── pifr_features.py          # 60-D PIFR feature extraction
│   ├── hybrid_fall_transformer.py # Transformer model
│   ├── types.py                  # FrameDiag dataclass
│   ├── viz.py                    # COCO keypoint edges
│   └── groups.py                 # Subject group ID extraction
├── scripts/
│   ├── augmentation.py           # Temporal shift, noise, flip
│   └── export_onnx.py            # ONNX export with validation
├── tests/                        # Unit tests
├── configs/                      # Config YAML files
├── data/                         # Processed data
├── AIO_Dataset/                  # Merged dataset
├── main.py                       # CLI + Tkinter GUI
├── gui_app.py                    # Tkinter GUI
├── app_inference.py              # PyQt5 GUI + inference
├── data_extractor.py             # AIO → X_train.npy keypoint extraction
├── train_transformer.py          # Training script
├── prepare_dataset.py             # URFD + GMDCSA preparation
├── prepare_le2i_dataset.py       # LE2I preparation
├── le2i_zone_based_extractor.py  # Zone-based LE2I extraction
├── evaluate.py                   # Full evaluation + visualizations
├── kaggle_train.py               # Kaggle full pipeline
├── kaggle_training_pipeline.ipynb # Kaggle notebook
├── yolo11n-pose.pt              # YOLOv11n-Pose pretrained weights
├── best_hybrid_transformer.pth    # Trained checkpoint
├── requirements.txt              # Dependencies
├── metadata.json                 # Training metadata
├── metrics.json                  # Evaluation metrics
├── groups.npy                    # Subject groups
├── README.md                     # Project documentation
└── PROJECT_REPORT.md             # Technical report
```

---

## 8. Key References & Methodology

- **YOLOv11-Pose**: Ultralytics YOLOv11n-Pose, COCO 17-keypoint format
- **Transformer**: Vaswani et al. (2017) — "Attention Is All You Need", NIPS
- **BCEWithLogitsLoss + AdamW**: Benabdennour et al. (2026), IEEE Access Fall Detection
- **Gaussian Noise σ=0.01**: PLOS ONE Fall Detection Studies (2026)
- **Zone-Based Protocol**: Original contribution, IEEE 2026 format

---

*This file was auto-generated from codebase analysis for Academic AI thesis writing assistance.*
*Last updated: May 14, 2026*
