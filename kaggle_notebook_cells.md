# Hybrid YOLOv11-Pose + Transformer — Hướng dẫn chạy Local

## Cách chạy đơn giản nhất

---

## Bước 1: Clone repository

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo-name>
```

---

## Bước 2: Cài dependencies

```bash
pip install -r requirements.txt
```

---

## Bước 3: Set dataset path (nếu có dataset)

```python
import os

# Dataset paths - sửa đường dẫn phù hợp với máy bạn
os.environ["FALL_DATASET_ROOT"] = "E:/datasets/fall-detection"
os.environ["FALL_WORK_ROOT"] = "E:/workspace/fall-detection"

print("FALL_DATASET_ROOT =", os.environ.get("FALL_DATASET_ROOT", "(default)"))
print("FALL_WORK_ROOT    =", os.environ.get("FALL_WORK_ROOT", "(default)"))
```

---

## Bước 4: Chạy pipeline

```bash
python -m src.kaggle_pipeline --strict
```

---

## Bước 5: Kiểm tra kết quả

```bash
python -m src.kaggle_sanity --strict
```

---

## Cấu trúc dataset mong đợi

```
FALL_DATASET_ROOT/
├── URFD/
│   └── (Fall|fall, ADL|adl)/*.zip
├── GMDCSA24/
│   └── Subject */(Fall|fall, ADL|adl)/*.mp4
└── LE2I/
    └── (Fall|fall, ADL|adl)/*.avi/*.mp4
```

## Ghi chú

1. **Checkpoint output:** `runs/best_hybrid_transformer.pth`
2. **Inference:** Sử dụng `app_inference.py` hoặc `gui_app.py` để test model
3. **GPU:** Khuyến nghị bật GPU để train nhanh hơn
