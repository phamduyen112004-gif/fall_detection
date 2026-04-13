# Hybrid YOLOv11-Pose + Transformer — Kaggle Notebook (6 cells)

Sao chép từng khối code bên dưới vào một notebook Kaggle theo thứ tự **CELL 1 → 6** trong cùng một kernel.

### Dữ liệu trong repo GitHub (clone trên Kaggle)

Nếu bạn đã commit dữ liệu dưới `data/raw/URFD` và `data/raw/GMDCSA24` (hoặc chỉ một trong hai), trên Kaggle:

1. Bật **Internet** (Settings). Trong **CELL 2** sửa `REPO_URL` (và `REPO_ROOT` nếu tên thư mục clone khác); code sẽ **`git clone`** tự động khi chưa có `REPO_ROOT`.
2. Sau clone, dữ liệu đọc từ `REPO_ROOT / "data/raw/URFD"` và `.../GMDCSA24` (hoặc bỏ qua nếu thiếu một phần).

### Dữ liệu qua Kaggle Input (zip riêng)

Bỏ comment **Cách B** trong CELL 2 và gán `URFD_ROOT` / `GMDCSA_ROOT` tới `/kaggle/input/...`.

---

## CELL 1: Setup & Imports

```python
# CELL 1: Setup & Imports
!pip install -q ultralytics

import os
import math
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm.auto import tqdm
from ultralytics import YOLO
```

---

## CELL 2: Data Preparation (All-In-One Dataset)

```python
# CELL 2: Chuẩn bị AIO_Dataset — dữ liệu trong repo (clone) hoặc Kaggle Input
import re
import shutil
import zipfile
from pathlib import Path

# --- Cách A: clone GitHub (bật Internet trong Settings). Sửa REPO_URL; REPO_ROOT = thư mục đích clone. ---
import subprocess

REPO_URL = "https://github.com/<USER>/<REPO>.git"  # <-- SỬA thành URL repo thật của bạn
REPO_ROOT = Path("/kaggle/working/fall-detection")  # phải trùng thư mục đích của lệnh clone bên dưới

if not REPO_ROOT.is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_ROOT)],
        check=True,
    )
    print("[git] clone xong:", REPO_ROOT)
else:
    print("[git] đã có sẵn, bỏ qua clone:", REPO_ROOT)

URFD_ROOT = REPO_ROOT / "data" / "raw" / "URFD"
GMDCSA_ROOT = REPO_ROOT / "data" / "raw" / "GMDCSA24"

# --- Cách B: dùng dataset đã thêm từ tab Input (bỏ comment và chỉnh tên dataset) ---
# URFD_ROOT = Path("/kaggle/input/urfd-dataset")
# GMDCSA_ROOT = Path("/kaggle/input/gmdcsa24-dataset")

if not REPO_ROOT.is_dir():
    print(
        f"[info] Chưa thấy REPO_ROOT={REPO_ROOT}. "
        "Nếu chỉ dùng Kaggle Input, hãy bật Cách B ở trên và gán lại URFD_ROOT / GMDCSA_ROOT."
    )

AIO_ROOT = Path("/kaggle/working/AIO_Dataset")
(AIO_ROOT / "fall").mkdir(parents=True, exist_ok=True)
(AIO_ROOT / "nofall").mkdir(parents=True, exist_ok=True)


def safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s or "clip"


def subject_slug(subject_dir: Path) -> str:
    m = re.search(r"(\d+)", subject_dir.name)
    return f"subject{m.group(1)}" if m else safe_stem(subject_dir.name).lower()


# URFD: giải nén zip -> fall / nofall (Fall/fall và ADL/adl)
if URFD_ROOT.is_dir():
    for src_names, aio_sub, tag in [
        (("Fall", "fall"), "fall", "fall"),
        (("ADL", "adl"), "nofall", "adl"),
    ]:
        src = next((URFD_ROOT / nm for nm in src_names if (URFD_ROOT / nm).is_dir()), None)
        if src is None:
            print(f"[skip URFD] không có: {URFD_ROOT}/({'|'.join(src_names)})")
            continue
        for zp in sorted(src.glob("*.zip")):
            stem = safe_stem(zp.stem)
            out_dir = AIO_ROOT / aio_sub / f"urfd_{tag}_{stem}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(out_dir)
            print(f"[URFD] {zp.name} -> {out_dir}")
else:
    print(f"[skip URFD] không có thư mục: {URFD_ROOT}")

# GMDCSA-24: copy mp4 (Subject */Fall|fall, */ADL|adl)
if GMDCSA_ROOT.is_dir():
    for subj_dir in sorted(GMDCSA_ROOT.iterdir()):
        if not subj_dir.is_dir():
            continue
        slug = subject_slug(subj_dir)
        for src_names, aio_sub in [(("Fall", "fall"), "fall"), (("ADL", "adl"), "nofall")]:
            act_dir = next((subj_dir / nm for nm in src_names if (subj_dir / nm).is_dir()), None)
            if act_dir is None:
                continue
            for vid in sorted(act_dir.glob("*.mp4")):
                new_name = f"gmdcsa_{slug}_{vid.stem}.mp4"
                dest = AIO_ROOT / aio_sub / new_name
                shutil.copy2(vid, dest)
                print(f"[GMDCSA] {vid.name} -> {dest}")
else:
    print(f"[skip GMDCSA] không có thư mục: {GMDCSA_ROOT}")

print("AIO_Dataset:", AIO_ROOT)
```

---

## CELL 3: Feature Extraction & Bug Fixes (CRUCIAL)

```python
# CELL 3: Feature Extraction & Bug Fixes (CRUCIAL)
import math
import re
from pathlib import Path

# --- Hằng số COCO 17 ---
NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

MIN_MEAN_CONF = 0.2
KPT_TH = 0.2
IMGSZ = 640
SEQ_LEN = 60
EPS = 1e-6

POSE_WEIGHTS = "yolo11n-pose.pt"
AIO_ROOT = Path("/kaggle/working/AIO_Dataset")
OUT_X = Path("/kaggle/working/X_train.npy")
OUT_Y = Path("/kaggle/working/y_train.npy")
OUT_G = Path("/kaggle/working/groups.npy")


def group_id_from_clip_path(path: Path) -> str:
    """Đồng bộ với src/groups.py — chia train/val theo subject / clip URFD."""
    name = path.name
    m = re.match(r"(gmdcsa_subject\d+)_", name, re.I)
    if m:
        return m.group(1).lower()
    m2 = re.match(r"^(urfd_(?:fall|adl)_[\w\-.]+)", name, re.I)
    if m2:
        return m2.group(1).lower()
    return str(path.resolve())


def angle_vertical(pt1: np.ndarray, pt2: np.ndarray) -> float:
    v = pt2.astype(np.float64) - pt1.astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return 0.0
    return float(math.acos(float(np.clip(v[1] / n, -1.0, 1.0))))


def angle_horizontal(pt1: np.ndarray, pt2: np.ndarray) -> float:
    v = pt2.astype(np.float64) - pt1.astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return 0.0
    return float(math.acos(float(np.clip(v[0] / n, -1.0, 1.0))))


def angle_at_b(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    ba = pa.astype(np.float64) - pb.astype(np.float64)
    bc = pc.astype(np.float64) - pb.astype(np.float64)
    n1, n2 = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
    if n1 < EPS or n2 < EPS:
        return 0.0
    c = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return float(math.acos(c))


def leg_angle(p_hip: np.ndarray, p_knee: np.ndarray, p_ankle: np.ndarray) -> float:
    v1 = p_knee.astype(np.float64) - p_hip.astype(np.float64)
    v2 = p_ankle.astype(np.float64) - p_knee.astype(np.float64)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < EPS or n2 < EPS:
        return 0.0
    return float(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def smart_mid_xy(
    k: np.ndarray, ia: int, ib: int, com_x: float, com_y: float
) -> tuple[float, float]:
    ca, cb = float(k[ia, 2]), float(k[ib, 2])
    xa, ya = float(k[ia, 0]), float(k[ia, 1])
    xb, yb = float(k[ib, 0]), float(k[ib, 1])
    if ca >= KPT_TH and cb >= KPT_TH:
        return (xa + xb) * 0.5, (ya + yb) * 0.5
    if ca >= KPT_TH:
        return xa, ya
    if cb >= KPT_TH:
        return xb, yb
    return com_x, com_y


def compute_geometry_9(kn: np.ndarray) -> np.ndarray:
    """9 PIFR: com_x, com_y, torso, nose_ankle, hip, shoulder, L_leg, R_leg, shoulder_nose — không bbox."""
    xy = kn[:, :2].astype(np.float64)
    com_x = float(np.mean(xy[:, 0]))
    com_y = float(np.mean(xy[:, 1]))

    nose = np.array([float(kn[NOSE, 0]), float(kn[NOSE, 1])], dtype=np.float64)
    mhx, mhy = smart_mid_xy(kn, L_HIP, R_HIP, com_x, com_y)
    mid_hip = np.array([mhx, mhy], dtype=np.float64)
    ax, ay = smart_mid_xy(kn, L_ANKLE, R_ANKLE, com_x, com_y)
    mid_ankle = np.array([ax, ay], dtype=np.float64)
    # mid_shoulder (smart) — đồng bộ quy tắc với mid_hip / mid_ankle
    _ = smart_mid_xy(kn, L_SHOULDER, R_SHOULDER, com_x, com_y)

    torso = angle_vertical(nose, mid_hip)
    nose_to_ankle = angle_vertical(nose, mid_ankle)

    if float(kn[L_HIP, 2]) >= KPT_TH and float(kn[R_HIP, 2]) >= KPT_TH:
        hip_ang = angle_horizontal(xy[L_HIP], xy[R_HIP])
    else:
        hip_ang = 0.0

    if float(kn[L_SHOULDER, 2]) >= KPT_TH and float(kn[R_SHOULDER, 2]) >= KPT_TH:
        shoulder_ang = angle_horizontal(xy[L_SHOULDER], xy[R_SHOULDER])
    else:
        shoulder_ang = 0.0

    def leg_ok(ih, ik, ia):
        return (
            float(kn[ih, 2]) >= KPT_TH
            and float(kn[ik, 2]) >= KPT_TH
            and float(kn[ia, 2]) >= KPT_TH
        )

    ll = leg_angle(xy[L_HIP], xy[L_KNEE], xy[L_ANKLE]) if leg_ok(L_HIP, L_KNEE, L_ANKLE) else 0.0
    rl = leg_angle(xy[R_HIP], xy[R_KNEE], xy[R_ANKLE]) if leg_ok(R_HIP, R_KNEE, R_ANKLE) else 0.0
    sh_nose = angle_at_b(xy[L_SHOULDER], xy[NOSE], xy[R_SHOULDER])

    return np.array(
        [com_x, com_y, torso, nose_to_ankle, hip_ang, shoulder_ang, ll, rl, sh_nose],
        dtype=np.float32,
    )


def frame_to_60(kn: np.ndarray) -> np.ndarray:
    g9 = compute_geometry_9(kn)
    return np.concatenate([kn.reshape(-1).astype(np.float32), g9], axis=0)


def resample_to_60(seq: np.ndarray) -> np.ndarray:
    t, f = seq.shape[0], seq.shape[1]
    if t == 0:
        return np.zeros((60, f), dtype=np.float32)
    if t == 1:
        return np.tile(seq.astype(np.float32), (60, 1))
    if t >= 60:
        idx = np.linspace(0, t - 1, 60, dtype=np.float64)
        idx = np.clip(np.round(idx).astype(np.int64), 0, t - 1)
        return seq[idx].astype(np.float32)
    pad = np.tile(seq[-1:], (60 - t, 1))
    return np.vstack([seq.astype(np.float32), pad])


def extract_from_bgr(frame_bgr: np.ndarray, model: YOLO) -> np.ndarray | None:
    frame_bgr = cv2.resize(frame_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    h, w = frame_bgr.shape[:2]
    results = model.predict(frame_bgr, imgsz=IMGSZ, verbose=False)
    if not results or results[0].keypoints is None or results[0].keypoints.data is None:
        return None
    r0 = results[0]
    kall = r0.keypoints.data.cpu().numpy()
    if kall.size == 0:
        return None
    best_i = int(np.argmax([float(k[:, 2].mean()) for k in kall]))
    k = kall[best_i].astype(np.float32)
    mean_c = float(k[:, 2].mean())
    if mean_c < MIN_MEAN_CONF:
        return None
    kn = k.copy()
    kn[:, 0] /= float(w)
    kn[:, 1] /= float(h)
    return frame_to_60(kn)


def process_clip(path: Path, model: YOLO) -> np.ndarray | None:
    """prev_vec chỉ trong một clip; reset mỗi lần gọi."""
    prev_vec: np.ndarray | None = None
    feats: list[np.ndarray] = []

    if path.is_dir():
        imgs = sorted(
            f
            for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        for fp in imgs:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            vec = extract_from_bgr(bgr, model)
            if vec is None:
                if prev_vec is not None:
                    feats.append(prev_vec.copy())
            else:
                feats.append(vec)
                prev_vec = vec.copy()
    elif path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            vec = extract_from_bgr(bgr, model)
            if vec is None:
                if prev_vec is not None:
                    feats.append(prev_vec.copy())
            else:
                feats.append(vec)
                prev_vec = vec.copy()
        cap.release()
    else:
        return None

    if not feats:
        return None
    return resample_to_60(np.stack(feats, axis=0))


def collect_sources() -> list[tuple[Path, int]]:
    out = []
    for lab, sub in [(1, "fall"), (0, "nofall")]:
        root = AIO_ROOT / sub
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir() or (p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}):
                out.append((p, lab))
    return out


model = YOLO(POSE_WEIGHTS)
pairs = collect_sources()
X_list: list[np.ndarray] = []
y_list: list[int] = []
g_list: list[str] = []

for p, lab in tqdm(pairs, desc="Clips"):
    if not p.exists():
        continue
    seq = process_clip(p, model)
    if seq is None or seq.shape != (60, 60):
        continue
    X_list.append(seq)
    y_list.append(lab)
    g_list.append(group_id_from_clip_path(p))
    del seq

if not X_list:
    raise RuntimeError("Không trích được mẫu nào. Kiểm tra AIO_Dataset và đường dẫn input.")

X_train = np.stack(X_list, axis=0).astype(np.float32)
y_train = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)
groups = np.array(g_list, dtype=object)
np.save(OUT_X, X_train)
np.save(OUT_Y, y_train)
np.save(OUT_G, groups, allow_pickle=True)
print("Saved", OUT_X, X_train.shape, OUT_Y, y_train.shape, OUT_G, groups.shape)
```

---

## CELL 4: Dataset Splitting & DataLoader

```python
# CELL 4: Dataset Splitting & DataLoader (ưu tiên GroupShuffleSplit như train_transformer.py)
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset

X = np.load("/kaggle/working/X_train.npy")
y = np.load("/kaggle/working/y_train.npy").reshape(-1)
g_path = Path("/kaggle/working/groups.npy")
VAL_RATIO = 0.2
SEED = 42

if g_path.is_file():
    groups = np.load(g_path, allow_pickle=True)
    if len(groups) == len(X):
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
            tr_idx, va_idx = next(gss.split(X, y, groups))
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            print("[split] GroupShuffleSplit (theo groups.npy)")
        except ValueError as e:
            print(f"[warn] GroupShuffleSplit thất bại ({e}); stratify theo nhãn.")
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=VAL_RATIO, random_state=SEED, stratify=y
            )
    else:
        print("[warn] len(groups) != len(X); stratify theo nhãn.")
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=VAL_RATIO, random_state=SEED, stratify=y
        )
else:
    print("[warn] Không có groups.npy; stratify theo nhãn.")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=VAL_RATIO, random_state=SEED, stratify=y
    )

X_tr_t = torch.from_numpy(X_tr).float()
y_tr_t = torch.from_numpy(y_tr).float().unsqueeze(1)
X_va_t = torch.from_numpy(X_va).float()
y_va_t = torch.from_numpy(y_va).float().unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    TensorDataset(X_va_t, y_va_t), batch_size=64, shuffle=False
)

print("Train", X_tr.shape, "Val", X_va.shape)
```

---

## CELL 5: Transformer Model Architecture

```python
# CELL 5: Transformer Model Architecture


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]


class HybridFallTransformer(nn.Module):
    def __init__(
        self,
        in_features: int = 60,
        seq_len: int = 60,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(in_features, d_model)
        self.scale = math.sqrt(float(d_model))
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) * self.scale
        h = h + self.pos_encoder(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h)
```

---

## CELL 6: Training Loop & Early Stopping

```python
# CELL 6: Training, Early Stopping & tuning ngưỡng (khớp train_transformer.py)
import copy

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridFallTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
crit = nn.BCEWithLogitsLoss()

best_f1 = -1.0
best_state = None
patience, bad = 25, 0
EPOCHS = 100

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    n = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        bs = xb.size(0)
        tr_loss += loss.item() * bs
        n += bs
    tr_loss /= max(n, 1)

    model.eval()
    va_logits, va_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            lg = model(xb).cpu().numpy()
            va_logits.append(lg.ravel())
            va_true.append(yb.numpy().ravel())
    y_logit = np.concatenate(va_logits)
    y_true_ep = np.concatenate(va_true)
    y_prob = 1.0 / (1.0 + np.exp(-y_logit))
    y_pred_05 = (y_prob >= 0.5).astype(np.int32)
    f1 = f1_score(y_true_ep, y_pred_05, zero_division=0)

    print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={tr_loss:.4f}  val_f1@0.5={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print(f"Early stop. Best val_f1@0.5={best_f1:.4f}")
            break

if best_state is None:
    best_state = model.state_dict()

model.load_state_dict(best_state)

model.eval()
va_logits, va_true = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        lg = model(xb).cpu().numpy()
        va_logits.append(lg.ravel())
        va_true.append(yb.numpy().ravel())
y_true = np.concatenate(va_true)
val_probs = 1.0 / (1.0 + np.exp(-np.concatenate(va_logits)))

best_thr, best_f1_thr = 0.5, -1.0
for t in np.linspace(0.05, 0.95, 91):
    f1t = float(f1_score(y_true, (val_probs >= t).astype(np.int32), zero_division=0))
    if f1t > best_f1_thr:
        best_f1_thr = f1t
        best_thr = float(t)

print("\n--- Validation (best weights, sau tinh chỉnh ngưỡng) ---")
print(f"best_threshold={best_thr:.4f}  val_f1={best_f1_thr:.4f}")
print("Confusion matrix [ [TN FP] [FN TP] ]:")
print(confusion_matrix(y_true.astype(np.int32), (val_probs >= best_thr).astype(np.int32)))
try:
    print(f"ROC-AUC: {roc_auc_score(y_true, val_probs):.4f}")
except ValueError:
    print("ROC-AUC: n/a")
try:
    print(f"PR-AUC: {average_precision_score(y_true, val_probs):.4f}")
except ValueError:
    print("PR-AUC: n/a")

SEQ_LEN = 60
FEATURE_DIM = 60
D_MODEL = 256

torch.save(
    {
        "model_state_dict": best_state,
        "best_val_f1": best_f1,
        "best_val_f1_tuned": best_f1_thr,
        "best_threshold": best_thr,
        "d_model": D_MODEL,
        "seq_len": SEQ_LEN,
        "in_features": FEATURE_DIM,
    },
    "/kaggle/working/best_hybrid_transformer.pth",
)
print("Saved /kaggle/working/best_hybrid_transformer.pth")
```

---

## Ghi chú

1. **CELL 1:** Trên một số notebook, nên cài `ultralytics` **trước** rồi `import`; nếu lỗi, tách thành ô chỉ `!pip install` rồi chạy lại kernel hoặc đặt `pip` ở đầu file.
2. **CELL 2:** Sửa `REPO_URL` và (nếu cần) `REPO_ROOT`; lần đầu ô sẽ `git clone` qua `subprocess`. Đường dữ liệu: `data/raw/URFD`, `data/raw/GMDCSA24`. **Git LFS:** sau clone có thể thêm ô `!git lfs install && cd ... && git lfs pull`. Chỉ Input: bật **Cách B** trong CELL 2.
3. **CELL 3:** Dòng `_ = smart_mid_xy(..., L_SHOULDER, R_SHOULDER, ...)` đảm bảo quy tắc **mid_shoulder** giống `mid_hip` / `mid_ankle` (đủ spec); góc **shoulder_nose** dùng 3 điểm vai–mũi–vai. File **`groups.npy`** dùng chung logic với `src/groups.py` để CELL 4 chia train/val theo nhóm.
4. **CELL 4 / 6:** Cách chia và lưu checkpoint (**`best_threshold`**, `best_val_f1_tuned`, …) khớp **`train_transformer.py`**; suy luận có thể dùng cùng file `.pth` với `app_inference.py`.
5. **Bộ nhớ:** Xây `X_list` theo từng clip; sau `np.stack` một lần. Có thể `del model` sau khi xong nếu cần RAM trước khi train.
