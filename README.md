# Hybrid YOLOv11-Pose + Transformer — Fall Detection

Hai hướng suy luận:

| Cách chạy | Mô tả | Lệnh |
|-----------|--------|------|
| **Pipeline góc + thời gian** | Tkinter, quy tắc tư thế nằm ngang + thời gian, không cần model đã học Transformer | `python main.py --gui` |
| **Transformer** | PyQt5, chuỗi 60×60 đặc trưng PIFR + `HybridFallTransformer`, cần `best_hybrid_transformer.pth` | `python main.py --gui-transformer` hoặc `python app_inference.py` |

**Chuẩn bị dữ liệu**

1. `python prepare_dataset.py` — gộp URFD + GMDCSA vào `AIO_Dataset/` (tùy đường dẫn). URFD: zip trong `Fall`/`fall` và `ADL`/`adl` dưới cùng một thư mục gốc, ví dụ `--urfd-root data/raw/URFD` (file kiểu `ADL/adl-13-cam0-rgb.zip`). GMDCSA-24: `Subject N/Fall`, `Subject N/ADL` hoặc `fall`/`adl`. Ví dụ GMDCSA: `--gmdcsa-root data/raw/GMDCSA24 --skip-urfd`.
2. `python data_extractor.py --aio-dir AIO_Dataset --out-dir data/processed` — sinh `X_train.npy`, `y_train.npy`, `groups.npy`.

**Huấn luyện**

```bash
python train_transformer.py --data-dir data/processed
```

Checkpoint lưu `best_threshold` (ngưỡng tối ưu F1 trên validation) — `app_inference.py` đọc để so sánh xác suất.

**Đặc trưng 60 chiều** (một nguồn: `src/pifr_features.py`): 51 keypoint + 9 hình học gồm `shoulder_nose_angle` (không dùng `bbox_aspect_ratio` trong vector).

**Chạy test**

```bash
pip install pytest scikit-learn
pytest tests/ -q
```

Trên GitHub, workflow `.github/workflows/ci.yml` chạy `pytest tests/` khi push hoặc PR vào `main` / `master`.

**Kaggle + dữ liệu trong repo:** trong `kaggle_notebook_cells.md`, CELL 2 mặc định dùng `REPO_ROOT` → `data/raw/URFD` và `data/raw/GMDCSA24` sau khi `git clone` vào `/kaggle/working/...` (bật Internet; chỉnh tên thư mục repo nếu khác).

**Export ONNX (tùy chọn)** — cần `pip install onnx`.

```bash
python scripts/export_onnx.py --weights best_hybrid_transformer.pth --out model.onnx
```
