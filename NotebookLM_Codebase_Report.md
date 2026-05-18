# System Architecture & Codebase Implementation Report
## Real-Time Fall Detection using YOLOv11-Pose and a Hybrid Transformer

**Project:** Fall Detection System  
**Author:** Master's Thesis  
**Date:** May 2026  
**Version:** 1.0

---

## Table of Contents

1. [System Pipeline Overview](#1-system-pipeline-overview)
2. [Mathematical Implementation of PIFR Features](#2-mathematical-implementation-of-pifr-features)
3. [Transformer Architecture](#3-transformer-architecture)
4. [Training Pipeline & Data Augmentation](#4-training-pipeline--data-augmentation)
5. [Real-Time Inference & GUI System](#5-real-time-inference--gui-system)
6. [Data Preprocessing Pipeline](#6-data-preprocessing-pipeline)
7. [Benchmark & Evaluation Pipeline](#7-benchmark--evaluation-pipeline)
8. [Configuration Management](#8-configuration-management)

---

## 1. System Pipeline Overview

### 1.1 Two-Stage Pipeline Architecture

The system implements a two-stage cascaded pipeline designed for real-time fall detection:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: KEYPOINT EXTRACTION                  │
│                                                                 │
│  Raw Video Frame (H × W × 3)  ──►  YOLOv11-Pose  ──►  17    │
│  (BGR Color Image)                     (Ultralytics)        Keypoints │
│                                                           (x, y, conf)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 2: FALL CLASSIFICATION                      │
│                                                                 │
│  17 Keypoints (17×3)  ──►  PIFR Feature Extractor  ──►  60D  │
│  (x, y, confidence)        (Geometric Engineering)        Feature │
│                              │                      Vector         │
│                              ▼                              │
│                         (60, 60) Matrix                       │
│                         (Temporal Window)                       │
│                              │                                 │
│                              ▼                                 │
│              HybridFallTransformer  ──►  Fall / No-Fall        │
│              (d_model=256, 3 layers, 4 heads)    (Probability) │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Temporal Sliding Window

To capture temporal dynamics, a **sliding window** of 60 consecutive frames is maintained. Each frame produces a 60-dimensional feature vector. The window therefore forms a **(60, 60) matrix** — 60 time steps × 60 features per time step — which serves as the input tensor to the Transformer.

### 1.3 Datasets

The system exclusively uses two benchmark datasets:

- **CaucaFall**: Multi-subject fall and daily activity videos. Videos are labeled by directory structure (`Subject.*/Action/*.avi`, where "fall" in the action name maps to label `1`).
- **MCFD (Multiple Cameras Fall Dataset)**: Multi-camera fall videos with CSV annotations specifying `[start, end]` frame segments and per-segment labels.

---

## 2. Mathematical Implementation of PIFR Features

**File:** `src/pifr_features.py`

PIFR stands for **Pose-based Image Feature Representation**. The implementation extracts a **60-dimensional feature vector** per frame:

- **51 dimensions**: The 17 COCO keypoints, each with `[x, y, confidence]` → 17 × 3 = 51
- **9 dimensions**: Geometric features derived from angular and positional relationships

### 2.1 COCO Keypoint Indexing

The 17 COCO keypoints used (with zero-based indices):

| Index | Keypoint Name      | Used In               |
|-------|-------------------|-----------------------|
| 0     | nose              | F3, F4, F9           |
| 5     | left_shoulder     | F3, F6                |
| 6     | right_shoulder    | F3, F6                |
| 11    | left_hip          | F4, F5, F7            |
| 12    | right_hip         | F4, F5, F8            |
| 13    | left_knee         | F7                    |
| 14    | right_knee        | F8                    |
| 15    | left_ankle        | F7, F9                |
| 16    | right_ankle       | F8, F9                |

### 2.2 The 9 Geometric Features (Mathematical Formulation)

All keypoints are normalized to `[0, 1]` range by dividing by frame width/height before feature computation.

#### F1 & F2: Center of Mass (Weighted by Confidence)

The center of mass is computed as a **confidence-weighted centroid**:

$$\bar{x} = \frac{\sum_{i=1}^{17} kp_i^x \cdot conf_i}{\sum_{i=1}^{17} conf_i}$$

$$\bar{y} = \frac{\sum_{i=1}^{17} kp_i^y \cdot conf_i}{\sum_{i=1}^{17} conf_i}$$

**Critical Safeguard:** The denominator sum of confidence values can be zero (e.g., all keypoints have `conf=0`). To prevent `ZeroDivisionError`, a small constant is added:

```python
# Code from src/pifr_features.py, lines 235-240
conf_sum: float = float(np.sum(keypoints[:, 2])) + 1e-8   # epsilon safeguard
features[0] = float(np.sum(keypoints[:, 0] * keypoints[:, 2])) / conf_sum  # F1: Center X
features[1] = float(np.sum(keypoints[:, 1] * keypoints[:, 2])) / conf_sum  # F2: Center Y
```

**Mathematical Justification:** `epsilon = 1e-8` is approximately 4 orders of magnitude smaller than the smallest detectable confidence (0.01), ensuring the perturbation to the normalized coordinates is negligible while guaranteeing numerical stability.

#### F3: Shoulder-Nose Angle

The angle between the two shoulder-to-nose vectors, capturing head tilt:

$$\theta_3 = \arccos\left(\frac{\vec{v_L} \cdot \vec{v_R}}{\|\vec{v_L}\| \cdot \|\vec{v_R}\|}\right)$$

Where:
- $\vec{v_L} = (shoulder_{left} - nose)$
- $\vec{v_R} = (shoulder_{right} - nose)$

#### F4: Torso Angle

The angle of the torso from vertical, measured as the angle between the nose-to-midhip vector and the vertical axis (Y-axis):

$$\theta_4 = \arccos\left(\frac{v_y}{\|v\|}\right)$$

Where $v = (midhip - nose)$ and $v_y$ is its Y-component. An angle of **0°** means the person is standing upright; **~90°** means lying flat.

#### F5: Hip Angle

The angle of the hip line (left-to-right hip vector) from the horizontal:

$$\theta_5 = \arccos\left(\frac{v_x}{\|v\|}\right)$$

Where $v = (hip_{right} - hip_{left})$ and $v_x$ is its X-component.

#### F6: Shoulder Angle

The angle of the shoulder line (left-to-right shoulder vector) from the horizontal:

$$\theta_6 = \arccos\left(\frac{v_x}{\|v\|}\right)$$

Where $v = (shoulder_{right} - shoulder_{left})$.

#### F7: Left Leg Angle (Hip-Knee-Ankle Chain)

The internal angle of the left leg at the knee joint:

$$\theta_7 = \arccos\left(\frac{\vec{u} \cdot \vec{w}}{\|\vec{u}\| \cdot \|\vec{w}\|}\right)$$

Where:
- $\vec{u} = (knee_{left} - hip_{left})$
- $\vec{w} = (ankle_{left} - knee_{left})$

#### F8: Right Leg Angle (Hip-Knee-Ankle Chain)

Analogous to F7 for the right leg:

$$\theta_8 = \arccos\left(\frac{\vec{u} \cdot \vec{w}}{\|\vec{u}\| \cdot \|\vec{w}\|}\right)$$

Where:
- $\vec{u} = (knee_{right} - hip_{right})$
- $\vec{w} = (ankle_{right} - knee_{right})$

#### F9: Nose-Ankle Angle

The angle between the nose-to-midankle vector and the vertical axis — a strong indicator of whether a person is upright or fallen:

$$\theta_9 = \arccos\left(\frac{v_y}{\|v\|}\right)$$

Where $v = (midankle - nose)$ and $midankle = \frac{ankle_{left} + ankle_{right}}{2}$.

### 2.3 ArcCosine Numerical Safeguard

All angle computations use `np.arccos` on the cosine of the angle. Due to floating-point rounding, the dot product ratio can slightly exceed `[-1, 1]`, which would produce `NaN`. The implementation clips the ratio:

```python
# Code from src/pifr_features.py, line 247 (inside _angle helper)
return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
```

This guarantees: `-1 ≤ ratio ≤ 1`, thus `0 ≤ arccos(ratio) ≤ π`.

### 2.4 60D Feature Vector Assembly

The full 60D vector is assembled as:

```
60D Vector = [F1..F51: Flattened 17×3 keypoints] + [F52: Center X] + [F53: Center Y]
           + [F54: Shoulder-Nose Angle] + [F55: Torso Angle] + [F56: Hip Angle]
           + [F57: Shoulder Angle] + [F58: Left Leg Angle] + [F59: Right Leg Angle]
           + [F60: Nose-Ankle Angle]
```

### 2.5 Frame-Level Fallback (No-Person Detection)

When YOLOv11-Pose detects **no person** in a frame (e.g., person exits frame or occlusion), the function returns `None`. The caller handles this by:

1. **In `data_prep.py`**: Using the previous frame's feature vector (temporal persistence)
2. **In `gui_app.py`**: Using the last known keypoints, stored in `self.prev_keypoints` (initialized as zeros)

```python
# gui_app.py, lines 356-360
if keypoints is None:
    # No person detected - use previous frame's keypoints padded
    keypoints = self.prev_keypoints.copy()
else:
    self.prev_keypoints = keypoints.copy()
```

---

## 3. Transformer Architecture

**File:** `src/hybrid_transformer.py`

### 3.1 Architecture Overview

The **HybridFallTransformer** is a Transformer Encoder-based model that processes the (60, 60) temporal feature matrix to produce a binary fall/no-fall classification.

```
Input: (batch, 60, 60) — 60 frames × 60 features per frame
         │
         ▼
┌────────────────────────────────────────────┐
│  Input Projection (Linear)                 │
│  60 → 256                                  │
│  (batch, 60, 256)                         │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Sinusoidal Positional Encoding            │
│  Adds temporal position information        │
│  (batch, 60, 256)                         │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Dropout (p=0.1)                          │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  TransformerEncoder (3 layers, 4 heads)    │
│  norm_first=True (Pre-LN)                 │
│  FFN dim = 256 × 4 = 1024                 │
│  (batch, 60, 256)                         │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Mean Pooling over Sequence Dimension      │
│  dim=1: (batch, 60, 256) → (batch, 256)  │
│  NO CLS token used                        │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  MLP Classification Head                  │
│  256 → 128 (Linear + LayerNorm + GELU)   │
│  128 → 32  (Linear + LayerNorm + GELU)   │
│  32 → 1    (Linear → logit output)       │
└────────────────────────────────────────────┘
         │
         ▼
Output: (batch, 1) — Fall probability logit
```

### 3.2 Exact PyTorch Configuration

| Parameter        | Value   | Code Reference                     |
|------------------|---------|------------------------------------|
| `input_dim`      | 60      | Default in `__init__`              |
| `num_frames`     | 60      | Default in `__init__`             |
| `d_model`        | **256** | `HybridFallTransformer.__init__`   |
| `nhead`          | **4**   | `HybridFallTransformer.__init__`  |
| `num_layers`     | **3**   | `HybridFallTransformer.__init__`  |
| `dropout`        | 0.1     | Default in `__init__`             |
| `dim_feedforward`| 1024    | `d_model * 4 = 256 * 4`           |
| `norm_first`     | True    | Pre-LayerNorm architecture         |

```python
# src/hybrid_transformer.py, lines 110-122
encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
    d_model=d_model,          # 256
    nhead=nhead,              # 4
    dim_feedforward=d_model * 4,  # 1024
    dropout=dropout,          # 0.1
    batch_first=True,         # (batch, seq, features)
    norm_first=True,          # Pre-LayerNorm: LayerNorm → Attention → Residual
)

self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers,    # 3
)
```

### 3.3 Mean Pooling (No CLS Token)

Unlike BERT-style classifiers that prepend a `[CLS]` token and use its output for classification, this implementation uses **Mean Pooling** over the entire sequence:

```python
# src/hybrid_transformer.py, line 165
x = x.mean(dim=1)  # Average across 60 time steps
```

**Mathematical Definition:**
$$\text{ pooled } = \frac{1}{T} \sum_{t=1}^{T} h_t$$

Where $T = 60$ is the number of frames, and $h_t \in \mathbb{R}^{256}$ is the output from the Transformer encoder at time step $t$. This aggregates information across the entire temporal window, capturing the overall pose dynamics rather than focusing on a single positional token.

**Academic Justification:** Mean Pooling is commonly preferred over CLS tokens for sequence-level classification tasks in vision transformers (e.g., ViT, TimeSformer), as it preserves information from all time steps. The CLS token in BERT was designed for token-level tasks (e.g., NER), not for temporal sequence aggregation.

### 3.4 MLP Classification Head

After mean pooling, a two-layer MLP processes the 256D pooled representation:

```python
# src/hybrid_transformer.py, lines 124-134
self.classifier: nn.Sequential = nn.Sequential(
    nn.Linear(d_model, 128),    # 256 → 128
    nn.LayerNorm(128),
    nn.GELU(),                   # Gaussian Error Linear Unit
    nn.Dropout(dropout),         # 0.1
    nn.Linear(128, 32),          # 128 → 32
    nn.LayerNorm(32),
    nn.GELU(),
    nn.Dropout(dropout),         # 0.1
    nn.Linear(32, 1),            # 32 → 1 (logit output)
)
```

The output is a **single scalar logit** (not a probability). Sigmoid activation is applied externally during inference:

```python
probability = torch.sigmoid(output).item()  # gui_app.py, line 419
```

### 3.5 Sinusoidal Positional Encoding

The `PositionalEncoding` module uses standard sinusoidal PE as described in *"Attention Is All You Need"* (Vaswani et al., 2017):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
# src/hybrid_transformer.py, lines 34-46
pe: torch.Tensor = torch.zeros(max_len, d_model)  # (60, 256)
position: torch.Tensor = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (60, 1)
div_term: torch.Tensor = torch.exp(
    torch.arange(0, d_model, 2, dtype=torch.float)
    * (-math.log(10000.0) / d_model)
)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0)  # (1, 60, 256) for broadcast with batch
self.register_buffer("pe", pe)  # Non-trainable, device-aware
```

### 3.6 Weight Initialization

All linear and LayerNorm weights are explicitly initialized:

```python
# src/hybrid_transformer.py, lines 138-147
def _init_weights(self) -> None:
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

---

## 4. Training Pipeline & Data Augmentation

**File:** `src/trainer.py`

### 4.1 Training Hyperparameters

All hyperparameters are centrally defined in `src/config.py` via the `TrainingConfig` frozen dataclass:

| Hyperparameter        | Value   | Academic Justification                           |
|-----------------------|---------|-------------------------------------------------|
| `d_model`             | 256     | Sufficient representational capacity             |
| `nhead`               | 4       | Balance between parallel attention heads          |
| `num_layers`          | 3       | Deep enough for temporal patterns, not overfitting|
| `dropout`             | 0.1     | Light regularization                             |
| `learning_rate`        | 5×10⁻⁴  | Standard for Transformer training                |
| `weight_decay`        | 1×10⁻⁵  | AdamW L2 regularization                          |
| `batch_size`          | 64      | GPU memory / accuracy trade-off                  |
| `epochs`              | 100     | With early stopping (patience=25)                |
| `early_stopping_patience` | 25   | Prevent overfitting on validation set            |

### 4.2 Data Split Strategy

The data is split using `sklearn.model_selection.train_test_split` with **stratification** to preserve class distribution:

```
Raw Data → train_test_split (80% train_val, 20% test, stratify=y)
                │
                ▼
        train_test_split (90% train, 10% val, stratify=y_train_val)
                │
                ▼
┌───────────────┬─────────────────┬────────────────┐
│  Train (72%)  │  Val (8%)       │  Test (20%)    │
│  augment=True │  augment=False  │  augment=False │
└───────────────┴─────────────────┴────────────────┘
```

```python
# src/trainer.py, lines 135-149
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=42, stratify=y  # TEST_SIZE=0.2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=config.VAL_SIZE, random_state=42, stratify=y_train_val
)  # VAL_SIZE=0.1

train_dataset = FallDataset(X_train, y_train, augment=True)   # ✓ Augmentation ON
val_dataset   = FallDataset(X_val,   y_val,   augment=False)  # ✓ Augmentation OFF
test_dataset  = FallDataset(X_test,  y_test,  augment=False)  # ✓ Augmentation OFF
```

### 4.3 Online Data Augmentation (Training Only)

The `FallDataset` class applies **online augmentation** during training — computed on-the-fly per batch, not pre-computed. This ensures augmentation diversity with every epoch.

#### 4.3.1 Gaussian Noise

```python
# src/trainer.py, lines 100-103
def _augment(self, x: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x) * TrainingConfig.NOISE_STD  # NOISE_STD = 0.01
    x = x + noise
```

**Mathematical Definition:** For each element $x_{t,f}$ in the (60, 60) feature matrix:
$$\tilde{x}_{t,f} = x_{t,f} + \mathcal{N}(0, \sigma^2) \quad \text{where } \sigma = 0.01$$

This adds small Gaussian perturbation to simulate sensor noise in pose estimation.

#### 4.3.2 Temporal Masking

```python
# src/trainer.py, lines 105-110
num_frames = x.shape[0]  # 60
num_mask = int(num_frames * TrainingConfig.MASK_RATIO)  # MASK_RATIO = 0.05 → 3 frames

if num_mask > 0:
    mask_indices = torch.randperm(num_frames)[:num_mask]  # Select 3 random frames
    x[mask_indices] = 0  # Zero out entire 60D feature vector for those frames
```

**Mathematical Definition:**
$$\tilde{x}_{t, :} = \mathbf{0} \quad \text{for } t \in M, \quad M \subset \{1, \ldots, 60\}, |M| = 3$$

This randomly masks 3 entire temporal steps (out of 60) to zero, forcing the Transformer to learn robust temporal patterns that don't depend on any single frame.

#### 4.3.3 Data Leakage Prevention

**CRITICAL:** Augmentation is **strictly isolated** to the training set only:

| Dataset | `augment` Flag | Gaussian Noise | Temporal Masking |
|---------|---------------|----------------|------------------|
| Train   | `True`        | ✓ Applied      | ✓ Applied        |
| Val     | `False`       | ✗ Not applied  | ✗ Not applied    |
| Test    | `False`       | ✗ Not applied  | ✗ Not applied    |

This is enforced in `create_data_loaders()` (lines 143-145) and verified in `FallDataset.__getitem__()` (line 95).

### 4.4 Loss Function

```python
# src/trainer.py, line 280
criterion = nn.BCEWithLogitsLoss()
```

`BCEWithLogitsLoss` combines a **Sigmoid** activation with **Binary Cross-Entropy** in a numerically stable way:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(\sigma(\hat{y}_i)) + (1 - y_i) \cdot \log(1 - \sigma(\hat{y}_i)) \right]$$

Where $\sigma$ is the sigmoid function and $\hat{y}_i$ is the raw model logit. By combining sigmoid + BCE into one class, PyTorch avoids the numerical instability of computing sigmoid separately.

### 4.5 Optimizer & Scheduler

```python
# src/trainer.py, lines 281-283
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5, verbose=True)
```

- **AdamW**: Adam with **decoupled weight decay** (Loshchilov & Hutter, 2017), preventing interference between weight decay and adaptive gradient estimation.
- **ReduceLROnPlateau**: Reduces learning rate by **50%** when validation F1-score plateaus for **10 epochs**. This allows fine-grained convergence in later epochs.

### 4.6 Early Stopping

```python
# src/trainer.py, lines 291-325
best_val_f1 = 0.0
patience_counter = 0

for epoch in range(config.EPOCHS):
    # ... training ...
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)  # Save best model
    else:
        patience_counter += 1

    if patience_counter >= config.EARLY_STOPPING_PATIENCE:  # 25
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break
```

The best model (with highest validation F1) is saved as `best_model.pth`. After training, this checkpoint is loaded for test evaluation.

### 4.7 Memory Management During Training

```python
# src/trainer.py, lines 333-335 (after each epoch)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

- `gc.collect()`: Forces Python's garbage collector to deallocate objects no longer in scope.
- `torch.cuda.empty_cache()`: Releases cached GPU memory back to the CUDA allocator, preventing gradual OOM accumulation.

### 4.8 Seed for Reproducibility

```python
# src/trainer.py, lines 255-258
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

---

## 5. Real-Time Inference & GUI System

**File:** `src/gui_app.py`

### 5.1 Multi-Threaded Architecture

The GUI uses **PyQt5's QThread** to separate the inference pipeline from the main GUI thread, preventing UI freezing:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MAIN GUI THREAD                                   │
│                                                                      │
│  • Renders video frames (QLabel with QPixmap)                        │
│  • Updates status labels, buttons, confidence display                │
│  • Handles user input (button clicks, window events)                  │
│  • Communicates ONLY via pyqtSignal / pyqtSlot                        │
└──────────────────────────────────────────────────────────────────────┘
                          ▲  Signals (frame_ready, status_update, etc.)
                          │
┌──────────────────────────────────────────────────────────────────────┐
│                 VIDEO INFERENCE THREAD (QThread)                     │
│                                                                      │
│  • cv2.VideoCapture: Reads frames from webcam/video/RTSP             │
│  • YOLOv11-Pose: Extracts 17 keypoints per frame                     │
│  • PIFR Feature Extractor: Computes 60D features                     │
│  • Sliding Window: Maintains deque(maxlen=60) of features            │
│  • HybridFallTransformer: Inference every 15 frames (stride=15)       │
│  • TelegramAlerter: Sends alerts with cooldown                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 PyQt5 Signal/Slot Mechanism

All cross-thread communication uses typed signals:

```python
# src/gui_app.py, lines 200-207
class VideoInferenceThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)       # Processed frame with skeleton
    status_update = pyqtSignal(str, str)       # Status text + color
    fps_update = pyqtSignal(float)             # Current FPS
    fall_detected = pyqtSignal(bool)           # Fall state changed
    buffer_update = pyqtSignal(int, int)       # Buffer fill level
    confidence_update = pyqtSignal(float)      # Fall probability
    error_occurred = pyqtSignal(str)           # Error message
```

Slots are decorated with `@pyqtSlot(type)`:

```python
# src/gui_app.py, lines 894-910
@pyqtSlot(np.ndarray)
def update_frame(self, frame: np.ndarray):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    scaled_pixmap = pixmap.scaled(
        self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    self.video_label.setPixmap(scaled_pixmap)
```

### 5.3 Temporal Sliding Window (deque)

```python
# src/gui_app.py, lines 224-225
self.feature_window = deque(maxlen=config.WINDOW_SIZE)  # maxlen=60
```

**Key properties:**
- `maxlen=60`: When the 61st frame arrives, the oldest frame is automatically evicted — **no index out-of-bounds possible**.
- `append()`: New 60D PIFR features are appended after each frame.
- No locking required: Only the inference thread writes to this deque.

### 5.4 Stride-Based Inference

```python
# src/gui_app.py, lines 375-393
self.frame_counter += 1

if (self.frame_counter % self.stride == 0 and        # Every 15th frame
    len(self.feature_window) >= self.config.WINDOW_SIZE):  # Window is full
    fall_detected = self.check_fall_detection(self.config.FALL_THRESHOLD)
    # ... Telegram alert logic ...
elif len(self.feature_window) >= self.config.WINDOW_SIZE:
    # Between stride intervals: maintain previous state
    fall_detected = self.current_probability > self.config.FALL_THRESHOLD
```

**Mathematical Efficiency:**
- Video runs at 30 FPS → 30 frames/second
- Inference runs every 15 frames → **2 inferences/second**
- This achieves real-time performance on edge devices while maintaining detection accuracy.

### 5.5 Dual-Threshold Alert System

Two separate thresholds control different behaviors:

| Threshold          | Value | Purpose                                    |
|--------------------|-------|--------------------------------------------|
| `FALL_THRESHOLD`   | 0.6   | Updates GUI status display (NORMAL/FALL)  |
| `ALERT_MIN_PROB`  | 0.18  | Triggers Telegram emergency alert         |

The **alert threshold (0.18)** is intentionally **lower** than the display threshold (0.6) to ensure that Telegram alerts are triggered **before** the GUI status turns red, providing early warning in real-time emergency scenarios.

### 5.6 Telegram Alert Cooldown

```python
# src/gui_app.py, lines 153-158 (TelegramAlert.send_alert)
current_time = time.time()
if current_time - self.last_alert_time < self.cooldown:  # cooldown=15s
    self.logger.debug(f"Alert skipped (cooldown: {self.cooldown}s)")
    return False
```

**Configuration:** `ALERT_COOLDOWN_SEC = 10.0` (from `src/config.py`)

This prevents **Telegram spam** — after a fall is detected and an alert is sent, the system ignores subsequent detections for at least 10 seconds, even if the probability remains high.

### 5.7 Resource Cleanup

```python
# src/gui_app.py, lines 484-495
@pyqtSlot()
def stop(self):
    self.running = False
    self.logger.info("Stopping inference thread...")

    # Explicitly release GPU memory
    self.yolo_model = None
    self.transformer_model = None
    if self.device is not None and self.device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
```

And in the main loop:

```python
# src/gui_app.py, lines 307-343
while self.running:
    ret, frame = cap.read()
    # ... process frame ...
cap.release()  # Always called, even on early break
self.logger.info("Inference stopped")
```

`cv2.VideoCapture.release()` is guaranteed to be called even when the loop exits early (via `break` or error), preventing resource leaks.

---

## 6. Data Preprocessing Pipeline

**File:** `src/data_prep.py`

### 6.1 Temporal Standardization

Every video produces a variable number of frames. The `standardize_temporal_dim()` function in `src/utils.py` normalizes all videos to the fixed **(60, 60) shape**:

```python
# src/utils.py, lines 92-128
def standardize_temporal_dim(video_features, target_frames=60, max_frames=120):
    # Step 1: Truncate to first 120 frames if longer
    if len(video_features) > max_frames:
        video_features = video_features[:max_frames]

    # Step 2: Subsample every 2nd frame (120 → 60, 60 → 30, etc.)
    video_features = video_features[::2]

    # Step 3: Pad with last frame if shorter than 60
    if len(video_features) < target_frames:
        last_frame = video_features[-1]
        padding = np.tile(last_frame, (target_frames - len(video_features), 1))
        video_features = np.vstack([video_features, padding])

    return video_features  # Shape: (60, 60)
```

**Example transformations:**

| Original Frames | After Truncation | After Subsampling `[::2]` | Padding Needed |
|-----------------|------------------|---------------------------|---------------|
| 120             | 120              | 60                        | 0             |
| 90              | 90               | 45                        | 15            |
| 60              | 60               | 30                        | 30            |
| 30              | 30               | 15                        | 45            |

### 6.2 CaucaFall Processing

- Recursively traverses `Subject.*/Action/*.avi` directory structure
- Labels derived from `Action` directory name: `"fall"` → label `1`, else `0`
- Processes entire video as one segment

### 6.3 MCFD Processing

- Reads annotations from `data_tuple3.csv` with `[chute, cam, start, end, label]`
- Extracts only the annotated segment `[start, end]` from each video
- Supports multi-camera setup (different `cam` values for same event)

### 6.4 Output Naming Convention

```
X_cauca_Subject.01_Walking.avi.npy  →  X_cauca_Subject_01_Walking.npy
y_cauca_Subject.01_Walking.avi.npy  →  y_cauca_Subject_01_Walking.npy

X_mcfd_c01_cam1_row0.npy
y_mcfd_c01_cam1_row0.npy
```

---

## 7. Benchmark & Evaluation Pipeline

**File:** `src/evaluator.py`

### 7.1 Metrics Computed

| Category     | Metrics                                                           |
|--------------|-------------------------------------------------------------------|
| Accuracy     | Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score   |
| Probabilistic| AUC-ROC, Average Precision (AP)                                   |
| Confusion    | TP, FP, FN, TN                                                    |
| YOLO mAP     | mAP@0.5, mAP@0.5:0.95                                            |
| Efficiency   | Parameters (M), Model Size (MB), GFLOPs, Latency (ms), FPS         |

### 7.2 Latency Benchmarking

```python
# src/evaluator.py, lines 272-292
# Warmup: 10 iterations to warm GPU cache
for i in range(min(10, len(X))):
    _ = model(torch.FloatTensor(X[i : i + 1]).to(device))

if torch.cuda.is_available():
    torch.cuda.synchronize()  # Ensure GPU finishes before timing

# Timed runs: 100 iterations
for i in range(min(num_runs, len(X))):
    start = time.perf_counter()
    _ = model(torch.FloatTensor(X[i : i + 1]).to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Synchronize for accurate GPU timing
    latencies.append((time.perf_counter() - start) * 1000.0)
```

`torch.cuda.synchronize()` is called before and after each timed iteration to ensure accurate GPU timing (CUDA kernels are asynchronous).

### 7.3 GFLOPs Computation

```python
# src/evaluator.py, lines 244-251
from thop import profile
dummy_input = torch.randn(input_shape).to(device)  # (1, 60, 60)
flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
return float(flops / 1e9)  # Convert to GFLOPs
```

---

## 8. Configuration Management

**File:** `src/config.py`

### 8.1 TrainingConfig (Frozen Dataclass)

```python
# src/config.py, lines 85-120
@dataclass(frozen=True)
class TrainingConfig:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    input_dim: int = 60        # TOTAL_FEATURES (51 + 9)
    num_frames: int = 60       # TARGET_FRAMES
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 25
    noise_std: float = 0.01     # Augmentation
    mask_ratio: float = 0.05   # Augmentation (5%)
    test_size: float = 0.2
    val_size: float = 0.1
```

The `frozen=True` flag prevents accidental modification of hyperparameters after instantiation, ensuring configuration immutability throughout the training run.

### 8.2 Key Detection & Inference Constants

| Constant                      | Value   | Purpose                                        |
|-------------------------------|---------|------------------------------------------------|
| `TARGET_FRAMES`               | 60      | Standardized temporal window size              |
| `MAX_FRAMES`                  | 120     | Maximum frames before truncation                |
| `KEYPOINT_DIM`                | 17      | COCO keypoints                                 |
| `KEYPOINT_FEATURES`           | 51      | 17 × 3                                         |
| `GEOMETRIC_FEATURES`          | 9       | PIFR angular/positional features               |
| `TOTAL_FEATURES`              | 60      | KEYPOINT_FEATURES + GEOMETRIC_FEATURES         |
| `ALERT_COOLDOWN_SEC`          | 10.0    | Minimum seconds between Telegram alerts        |
| `ALERT_MIN_PROB`              | 0.18    | Probability threshold for Telegram alert       |

---

## Appendix: File Index

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `src/config.py`       | All hyperparameters, paths, dataclasses      |
| `src/pifr_features.py`| Keypoint extraction + 60D PIFR features     |
| `src/hybrid_transformer.py` | Transformer model architecture          |
| `src/trainer.py`      | Training loop, augmentation, early stopping  |
| `src/data_prep.py`    | Video preprocessing, dataset loading         |
| `src/evaluator.py`    | Benchmark, metrics, visualizations           |
| `src/gui_app.py`      | PyQt5 GUI, real-time inference, Telegram    |
| `src/utils.py`        | Helpers: device, metrics, temporal std.      |
| `main.py`             | Single entry point for all modes             |

---

*End of Report — Generated for AI Academic Review (NotebookLM)*
