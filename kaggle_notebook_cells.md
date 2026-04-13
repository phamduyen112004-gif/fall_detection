# Hybrid YOLOv11-Pose + Transformer — Kaggle Notebook (cách chạy đơn giản)

Nếu bạn muốn chạy “gọn” như bạn đề xuất (clone repo + 1 lệnh), hãy dùng **6 cell** dưới đây.

Yêu cầu: bật Internet (để clone repo và cài thư viện), và đã add dataset trong tab **Input**.

---

## CELL 1: Clone repository

```python
GIT_URL = "https://github.com/<username>/<repo>.git"  # <-- đổi URL repo của bạn
REPO = GIT_URL.rstrip("/").split("/")[-1].replace(".git", "")

%cd /kaggle/working
!rm -rf "/kaggle/working/{REPO}"
!git clone "{GIT_URL}"
%cd "/kaggle/working/{REPO}"
print("Cloned:", REPO)
```

---

## CELL 2: Install dependencies

```python
%cd "/kaggle/working/{REPO}"
!pip -q install -r requirements.txt
```

---

## CELL 3: Optional env override (dataset path)

```python
import os

# Dataset name theo ảnh bạn gửi:
# os.environ["FALL_DATASET_ROOT"] = "/kaggle/input/fall-detection-dataset"
# os.environ["FALL_WORK_ROOT"] = "/kaggle/working"

print("FALL_DATASET_ROOT =", os.environ.get("FALL_DATASET_ROOT", "(default in src.kaggle_pipeline)"))
print("FALL_WORK_ROOT    =", os.environ.get("FALL_WORK_ROOT", "(default: /kaggle/working)"))
```

---

## CELL 4: Run full pipeline

```python
%cd "/kaggle/working/{REPO}"
!python -m src.kaggle_pipeline --strict
```

---

## CELL 5: Sanity check outputs

```python
%cd "/kaggle/working/{REPO}"
!python -m src.kaggle_sanity --strict
```

---

## CELL 6 (tuỳ chọn): Optional ablation study (rule-based)

```python
%cd "/kaggle/working/{REPO}"
!python -m src.eval.ablation_runner --help
```

---

## Ghi chú

1. **CELL 1:** Bật **Internet** để `git clone` chạy được.
2. **CELL 2:** Nếu `pip install -r requirements.txt` lỗi do môi trường Kaggle, thử “Restart session” rồi chạy lại CELL 2.
3. **CELL 3:** Nếu dataset Input của bạn không phải `/kaggle/input/fall-detection-dataset`, hãy set `FALL_DATASET_ROOT` đúng tên. Cấu trúc mong đợi:
   - `FALL_DATASET_ROOT/URFD/(Fall|fall, ADL|adl)/*.zip`
   - `FALL_DATASET_ROOT/GMDCSA24/Subject */(Fall|fall, ADL|adl)/*.mp4`
4. **CELL 4:** Khuyến nghị bật **GPU** để trích đặc trưng và train nhanh hơn. Trong log của CELL 4 sẽ có thống kê nhanh: `N`, `fall/nofall`, `X_shape` và `groups unique`. Checkpoint xuất ra: `/kaggle/working/best_hybrid_transformer.pth`.
