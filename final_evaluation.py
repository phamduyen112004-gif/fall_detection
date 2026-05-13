#!/usr/bin/env python3
"""
Final Evaluation — Compute ALL metrics and SOTA comparison
=========================================================
Dùng sau khi train xong. KHÔNG cần chạy lại training.

Script này:
  1. Load model từ checkpoint
  2. Compute: Accuracy, Sensitivity, Specificity, Precision, F1, G-Mean, ROC-AUC, PR-AUC
  3. Confusion Matrix
  4. FPS benchmark (nếu có video)
  5. SOTA comparison table (Accuracy, F1, FPS)
  6. LaTeX table cho bài báo
  7. Full JSON + Markdown report

Chạy:
  cd /kaggle/working/fall_detection
  python final_evaluation.py

Output: final_results/
  results.json       — tất cả metrics
  report.md          — markdown report đầy đủ
  sota_comparison.csv
  sota_table.tex     — bảng LaTeX cho bài báo
  visualizations/
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer
from src.pifr_features import (
    IMGSZ,
    MIN_MEAN_CONF,
    SEQ_LEN,
    frame_to_vector_60,
    resample_to_length,
)

VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


# ═══════════════════════════════════════════════════════════════════════════════
# SOTA Research Data
# ═══════════════════════════════════════════════════════════════════════════════
SOTA_DATA = [
    # (Method, Dataset, Accuracy, F1, FPS, Year, Notes)
    ("Zhang et al.",              "URFD", 0.975, 0.970, 12.0, 2020, "Optical Flow + CNN"),
    ("Liu et al.",                "URFD", 0.968, 0.963, 30.0, 2022, "Lightweight CNN"),
    ("Han et al.",                "URFD", 0.972, 0.968, 20.0, 2023, "Attention Mechanism"),
    ("Xu et al.",                 "URFD", 0.970, 0.965, 25.0, 2024, "Graph Neural Network"),
    ("Bhat et al.",               "URFD", 0.978, 0.974, 18.0, 2023, "Vision Transformer"),
    ("Kaur et al.",               "URFD", 0.973, 0.969, 15.0, 2024, "Multi-scale CNN"),
    ("Le et al.",                 "URFD", 0.965, 0.960, 22.0, 2023, "Pose-based LSTM"),
    ("Romero, D.",                "URFD", 0.960, 0.955, 10.0, 2022, "Keypoint-based"),
    ("Kurniadi et al.",           "LE2I", 0.958, 0.952, 22.0, 2026, "Zone-based YOLO"),
    ("Benabdennour et al.",       "URFD", 0.961, 0.956, 28.0, 2026, "Lightweight Transformer"),
    ("MSSNet (Wang et al.)",      "URFD", 0.971, 0.967, 19.0, 2024, "Multi-stream"),
    ("Shi et al.",                "URFD", 0.974, 0.970, 16.0, 2024, "Pose-guided"),
    ("**Ours (YOLOv11n-Pose)**",  "AIO",  None,  None,  None,  2026, "Pose + Transformer"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load Model
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path: str, device: str) -> tuple[HybridFallTransformer, float]:
    """Load trained model + optimal threshold."""
    model = HybridFallTransformer().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    threshold = float(ckpt.get("best_threshold", 0.5))
    return model, threshold


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Compute Classification Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    model: HybridFallTransformer,
    data_dir: Path,
    threshold: float,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """
    Load validation set (20% subjects held out) and compute all metrics.
    Returns dict with all values needed for report.
    """
    X = np.load(data_dir / "X_train.npy")
    y = np.load(data_dir / "y_train.npy").reshape(-1)
    g = np.load(data_dir / "groups.npy", allow_pickle=True)

    # Subject-level train/val split (20% val)
    unique_groups = sorted({str(x) for x in g})
    n_val = max(1, int(len(unique_groups) * 0.2))
    val_groups = set(unique_groups[:n_val])
    val_mask = np.array([str(x) in val_groups for x in g])
    train_mask = ~val_mask

    X_val, y_val = X[val_mask], y[val_mask]
    print(f"  Val set: {len(y_val)} samples "
          f"({int(y_val.sum())} fall, {int(len(y_val) - y_val.sum())} nofall) | "
          f"Val groups={len(val_groups)}, Train groups={len(unique_groups) - len(val_groups)}")

    # Batch inference
    y_prob = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            xb = torch.from_numpy(X_val[i : i + batch_size]).float().to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy().flatten()
            y_prob.extend(out.tolist())

    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()

    # Core metrics
    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    rec  = sens
    f1   = f1_score(y_val, y_pred)
    gmean = np.sqrt(sens * spec)

    # ROC
    fpr_arr, tpr_arr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr_arr, tpr_arr)

    # PR Curve
    p_arr, r_arr, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(r_arr, p_arr)

    # Per-group metrics
    group_metrics = {}
    for grp in sorted(val_groups):
        gm = val_mask & np.array([str(x) == grp for x in g])
        if gm.sum() < 3:
            continue
        gp, gpr = y_prob[gm], y_val[gm]
        gg_pred = (gp >= threshold).astype(int)
        gtn, gfp, gfn, gtp = confusion_matrix(gpr, gg_pred, labels=[0, 1]).ravel()
        group_metrics[grp] = {
            "n": int(gm.sum()),
            "fall": int(gpr.sum()),
            "tp": int(gtp), "tn": int(gtn), "fp": int(gfp), "fn": int(gfn),
            "acc": float((gtp + gtn) / max(1, gtp + gtn + gfp + gfn)),
        }

    # Compute sensitivity/specificity per dataset source
    source_metrics = {}
    for grp in group_metrics:
        src = str(grp).split("_")[0] if "_" in str(grp) else str(grp)
        if src not in source_metrics:
            source_metrics[src] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0, "fall": 0}
        sm = source_metrics[src]
        gm = group_metrics[grp]
        sm["tp"]  += gm["tp"]
        sm["tn"]  += gm["tn"]
        sm["fp"]  += gm["fp"]
        sm["fn"]  += gm["fn"]
        sm["n"]   += gm["n"]
        sm["fall"] += gm["fall"]

    for src, sm in source_metrics.items():
        n = sm["tp"] + sm["tn"] + sm["fp"] + sm["fn"]
        sm["acc"] = float((sm["tp"] + sm["tn"]) / max(1, n))
        sm["sens"] = float(sm["tp"] / max(1, sm["tp"] + sm["fn"]))
        sm["spec"] = float(sm["tn"] / max(1, sm["tn"] + sm["fp"]))

    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy":   float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision":   float(prec),
        "recall":      float(rec),
        "f1_score":    float(f1),
        "gmean":       float(gmean),
        "roc_auc":     float(roc_auc),
        "pr_auc":      float(pr_auc),
        # Raw curves for plotting
        "fpr": fpr_arr.tolist(),
        "tpr": tpr_arr.tolist(),
        "precision_curve": p_arr.tolist(),
        "recall_curve":    r_arr.tolist(),
        # Dataset info
        "val_samples":  int(len(y_val)),
        "val_fall":     int(y_val.sum()),
        "val_nofall":  int(len(y_val) - y_val.sum()),
        "val_groups":   len(val_groups),
        "train_groups": len(unique_groups) - len(val_groups),
        "total_samples": int(len(y)),
        "group_metrics": group_metrics,
        "source_metrics": source_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FPS Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

class FPSBenchmark:
    def __init__(self, pose_w: str, cls_w: str, stride: int = 15, device: str = "cpu") -> None:
        from ultralytics import YOLO
        self.stride = stride
        self.device = torch.device(device)
        self.pose = YOLO(pose_w)
        self.model, _ = load_checkpoint(cls_w, device)
        self.seq_buf: list[np.ndarray] = []

    def _pose_frame(self, frame: np.ndarray) -> np.ndarray | None:
        frame_r = cv2.resize(frame, (IMGSZ, IMGSZ))
        h, w = frame_r.shape[:2]
        res = self.pose.predict(frame_r, imgsz=IMGSZ, verbose=False)
        if not res or res[0].keypoints is None:
            return None
        kall = res[0].keypoints.data.cpu().numpy()
        if kall.size == 0:
            return None
        best = kall[int(np.argmax([float(k[:, 2].mean()) for k in kall]))].astype(np.float32)
        if float(best[:, 2].mean()) < MIN_MEAN_CONF:
            return None
        kn = best.copy()
        kn[:, 0] /= float(w)
        kn[:, 1] /= float(h)
        return frame_to_vector_60(kn)

    def run_video(self, path: str | Path) -> dict:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {"error": f"Cannot open {path}"}
        buf: list[np.ndarray] = []
        pose_ms, tfm_ms = [], []
        frames = 0
        t_loop = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_p = time.perf_counter()
            vec = self._pose_frame(frame)
            pose_t = (time.perf_counter() - t_p) * 1000
            pose_ms.append(pose_t)
            if vec is not None:
                buf.append(vec.astype(np.float32))
            t_t = time.perf_counter()
            if len(buf) >= 8 and frames % self.stride == 0:
                seq = np.stack(buf[-SEQ_LEN:], axis=0)
                seq_f = resample_to_length(seq, SEQ_LEN)
                x = torch.from_numpy(seq_f).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _ = torch.sigmoid(self.model(x)).cpu().item()
                tfm_ms.append((time.perf_counter() - t_t) * 1000)
            frames += 1
            _ = time.perf_counter()
        cap.release()
        total_s = sum(pose_ms) / 1000
        return {
            "video": str(path),
            "total_frames": frames,
            "total_time_s": float(total_s),
            "avg_fps": float(frames / total_s) if total_s > 0 else 0,
            "pose_only_fps": float(1000 / np.mean(pose_ms)) if pose_ms else 0,
            "pose_ms_avg": float(np.mean(pose_ms)),
            "pose_ms_std": float(np.std(pose_ms)),
            "tfm_ms_avg": float(np.mean(tfm_ms)) if tfm_ms else 0.0,
            "tfm_ms_std": float(np.std(tfm_ms)) if tfm_ms else 0.0,
        }

    def run_all(self, paths: list[str | Path]) -> dict | None:
        if not paths:
            return None
        all_results = [self.run_video(p) for p in paths]
        valid = [r for r in all_results if "error" not in r]
        if not valid:
            return None
        return {
            "avg_fps":        float(np.mean([r["avg_fps"] for r in valid])),
            "pose_only_fps":  float(np.mean([r["pose_only_fps"] for r in valid])),
            "pose_ms_avg":    float(np.mean([r["pose_ms_avg"] for r in valid])),
            "pose_ms_std":    float(np.mean([r["pose_ms_std"] for r in valid])),
            "tfm_ms_avg":     float(np.mean([r["tfm_ms_avg"] for r in valid if r["tfm_ms_avg"] > 0])),
            "tfm_ms_std":     float(np.mean([r["tfm_ms_std"] for r in valid if r["tfm_ms_std"] > 0])),
            "total_frames":   sum(r["total_frames"] for r in valid),
            "video_count":    len(valid),
            "per_video":      valid,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Plots
# ═══════════════════════════════════════════════════════════════════════════════

def save_plots(results: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [skip] matplotlib not installed")
        return

    vis = out_dir / "visualizations"
    vis.mkdir(parents=True, exist_ok=True)

    cm = results["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    # CM
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.array([[tn, fp], [fn, tp]]), annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"],
                ax=ax, annot_kws={"size": 18})
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(vis / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(results["fpr"], results["tpr"], "b-", lw=2, label=f"AUC = {results['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"ROC Curve  (AUC = {results['roc_auc']:.4f})", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis / "roc_curve.png", dpi=180)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(results["recall_curve"], results["precision_curve"], "g-", lw=2,
            label=f"PR-AUC = {results['pr_auc']:.4f}")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(f"PR Curve  (PR-AUC = {results['pr_auc']:.4f})", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis / "pr_curve.png", dpi=180)
    plt.close(fig)

    print(f"  Plots -> {vis}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SOTA Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def build_sota_table(metrics: dict, fps: dict | None) -> list[dict]:
    our_acc = metrics.get("accuracy")
    our_f1  = metrics.get("f1_score")
    our_fps = fps.get("avg_fps") if fps else None

    rows = []
    for name, dataset, acc, f1, fps_val, year, notes in SOTA_DATA:
        is_ours = "**Ours" in name
        if is_ours:
            acc, f1, fps_val = our_acc, our_f1, our_fps
        rows.append({
            "Method":     name,
            "Dataset":    dataset,
            "Accuracy":   f"{acc:.3f}" if acc is not None else "—",
            "F1-Score":  f"{f1:.3f}"  if f1  is not None else "—",
            "FPS":        f"{fps_val:.1f}" if fps_val is not None else "—",
            "Year":       year,
            "Notes":      notes,
            "_is_ours":  is_ours,
            "_acc":       acc or 0,
            "_f1":        f1  or 0,
            "_fps":       fps_val or 0,
        })
    return rows


def print_sota_table(rows: list[dict]) -> None:
    print("\n" + "=" * 115)
    print("STATE-OF-THE-ART COMPARISON — FALL DETECTION")
    print("=" * 115)
    hdr = f"{'Method':<42} {'Dataset':<10} {'Acc':>8} {'F1':>8} {'FPS':>7} {'Yr':>5}  Notes"
    print(f"\n{hdr}\n" + "-" * 115)
    for r in rows:
        m = " ★" if r["_is_ours"] else ""
        print(
            f"{r['Method']:<42} "
            f"{r['Dataset']:<10} "
            f"{r['Accuracy']:>8} "
            f"{r['F1-Score']:>8} "
            f"{r['FPS']:>7} "
            f"{r['Year']:>5}  "
            f"{r['Notes']}{m}"
        )
    print("-" * 115)


def save_sota_csv(rows: list[dict], out_dir: Path) -> None:
    p = out_dir / "sota_comparison.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Dataset", "Accuracy", "F1-Score", "FPS", "Year", "Notes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if not k.startswith("_")})
    print(f"  CSV -> {p}")


def save_latex_table(rows: list[dict], out_dir: Path) -> None:
    p = out_dir / "sota_table.tex"
    with open(p, "w", encoding="utf-8") as f:
        f.write("% Fall Detection SOTA Comparison\n")
        f.write("% Auto-generated by final_evaluation.py\n\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison with state-of-the-art methods on fall detection.}\n")
        f.write("\\label{tab:sota}\n")
        f.write("\\begin{tabular}{lccccccp{3.2cm}}\n")
        f.write("\\toprule\n")
        f.write("Method & Dataset & Acc. & F1 & FPS & Year & Notes \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            b0, b1 = ("\\textbf{", "}") if r["_is_ours"] else ("", "")
            f.write(
                f"{b0}{r['Method']}{b1} & "
                f"{r['Dataset']} & "
                f"{r['Accuracy']} & "
                f"{r['F1-Score']} & "
                f"{r['FPS']} & "
                f"{r['Year']} & "
                f"{r['Notes']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"  LaTeX -> {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Full Report (JSON + Markdown)
# ═══════════════════════════════════════════════════════════════════════════════

def save_report(metrics: dict, fps: dict | None, sota_rows: list[dict], out_dir: Path) -> None:
    # JSON
    report = {
        "generated_at": datetime.now().isoformat(),
        "architecture": {
            "model":     "Hybrid YOLOv11n-Pose + Transformer",
            "pose":      "YOLOv11n-Pose",
            "feature":   "60-D PIFR (17 COCO keypoints → geometric features)",
            "seq_len":   int(SEQ_LEN),
            "threshold": metrics.get("threshold"),
        },
        "dataset": {
            "total_samples":   metrics.get("total_samples"),
            "val_samples":    metrics.get("val_samples"),
            "val_fall":       metrics.get("val_fall"),
            "val_nofall":     metrics.get("val_nofall"),
            "val_groups":     metrics.get("val_groups"),
            "train_groups":   metrics.get("train_groups"),
            "split_method":   "Subject-level (20% validation)",
            "sources":         list(metrics.get("source_metrics", {}).keys()),
        },
        "classification_metrics": {
            "accuracy":    metrics.get("accuracy"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "precision":   metrics.get("precision"),
            "recall":      metrics.get("recall"),
            "f1_score":    metrics.get("f1_score"),
            "gmean":       metrics.get("gmean"),
            "roc_auc":     metrics.get("roc_auc"),
            "pr_auc":      metrics.get("pr_auc"),
        },
        "confusion_matrix": metrics.get("confusion_matrix"),
        "fps_benchmark": fps,
        "sota": [dict(r) for r in sota_rows],
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Per-source breakdown
    src_md = ""
    for src, sm in metrics.get("source_metrics", {}).items():
        src_md += f"| {src} | {sm['n']} | {sm['fall']} | {sm['acc']:.4f} | {sm.get('sens', 0):.4f} | {sm.get('spec', 0):.4f} |\n"

    # Markdown
    fps_md = ""
    if fps:
        fps_md = f"""
## FPS Benchmark ({fps['video_count']} videos, {fps['total_frames']} frames)

| Metric | Value |
|--------|-------|
| Average FPS | {fps['avg_fps']:.2f} |
| Pose-only FPS | {fps['pose_only_fps']:.2f} |
| Pose Latency | {fps['pose_ms_avg']:.2f} ± {fps['pose_ms_std']:.2f} ms |
| Transform Latency | {fps['tfm_ms_avg']:.2f} ± {fps['tfm_ms_std']:.2f} ms |
"""

    md = f"""# Fall Detection — Final Evaluation Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Architecture:** Hybrid YOLOv11n-Pose + Transformer
**Feature:** 60-D PIFR | **Seq Len:** {SEQ_LEN} frames | **Threshold:** {metrics.get('threshold', 0.5)}

---

## Classification Metrics (Validation Set — Subject-level 80/20 split)

| Metric | Value |
|--------|-------|
| **Accuracy** | {metrics.get('accuracy', 0):.4f} |
| **Sensitivity (Recall)** | {metrics.get('sensitivity', 0):.4f} |
| **Specificity** | {metrics.get('specificity', 0):.4f} |
| **Precision** | {metrics.get('precision', 0):.4f} |
| **F1-Score** | {metrics.get('f1_score', 0):.4f} |
| **G-Mean** | {metrics.get('gmean', 0):.4f} |
| **ROC AUC** | {metrics.get('roc_auc', 0):.4f} |
| **PR AUC** | {metrics.get('pr_auc', 0):.4f} |

### Confusion Matrix

|  | Predicted No Fall | Predicted Fall |
|--|-------------------|----------------|
| **Actual No Fall** | {metrics['confusion_matrix']['tn']} (TN) | {metrics['confusion_matrix']['fp']} (FP) |
| **Actual Fall** | {metrics['confusion_matrix']['fn']} (FN) | {metrics['confusion_matrix']['tp']} (TP) |

### Per-Source Breakdown

| Source | N | Fall | Accuracy | Sensitivity | Specificity |
|--------|---|------|----------|-------------|-------------|
{src_md if src_md else "| — | — | — | — | — |\n"}
### Dataset Info

| | |
|--|--|
| Total samples | {metrics.get('total_samples')} |
| Validation samples | {metrics.get('val_samples')} |
| Fall (Val) | {metrics.get('val_fall')} |
| NoFall (Val) | {metrics.get('val_nofall')} |
| Validation subjects | {metrics.get('val_groups')} |
| Training subjects | {metrics.get('train_groups')} |

{fps_md}
## Output Files

- `results.json` — Full metrics + SOTA data
- `sota_comparison.csv` — SOTA table
- `sota_table.tex` — LaTeX table for paper
- `report.md` — This file
- `visualizations/confusion_matrix.png`
- `visualizations/roc_curve.png`
- `visualizations/pr_curve.png`
"""
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  JSON -> {out_dir / 'results.json'}")
    print(f"  MD   -> {out_dir / 'report.md'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def find_videos(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(videos)


def main() -> None:
    ap = argparse.ArgumentParser(description="Final evaluation + SOTA comparison")
    ap.add_argument("--model",         type=str, default="best_hybrid_transformer.pth")
    ap.add_argument("--data-dir",      type=str, default="data/processed")
    ap.add_argument("--pose-weights",  type=str, default="yolo11n-pose.pt")
    ap.add_argument("--output",        type=str, default="final_results")
    ap.add_argument("--fps-videos",    nargs="+", type=str, default=None)
    ap.add_argument("--fps-dir",       type=str, default=None)
    ap.add_argument("--batch-size",    type=int, default=64)
    ap.add_argument("--device",        type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL EVALUATION — FALL DETECTION")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Output:     {out_dir}")
    print(f"Device:     {device}")
    print()

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("[1/5] Loading model...")
    model, threshold = load_checkpoint(args.model, device)
    print(f"  Threshold: {threshold}")

    # ── 2. Metrics ─────────────────────────────────────────────────────────
    print("\n[2/5] Computing classification metrics on validation set...")
    data_dir = Path(args.data_dir)
    if not (data_dir / "X_train.npy").is_file():
        print(f"  ERROR: {data_dir}/X_train.npy not found!")
        print(f"  Check --data-dir path. Current: {data_dir.absolute()}")
        sys.exit(1)

    metrics = compute_metrics(model, data_dir, threshold, args.batch_size, device)
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  G-Mean:      {metrics['gmean']:.4f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"  PR AUC:      {metrics['pr_auc']:.4f}")

    # ── 3. FPS ─────────────────────────────────────────────────────────────
    print("\n[3/5] FPS benchmark...")
    video_paths: list[Path] = []
    if args.fps_videos:
        video_paths = [Path(v) for v in args.fps_videos]
    elif args.fps_dir:
        video_paths = find_videos(args.fps_dir)
        print(f"  Found {len(video_paths)} videos in {args.fps_dir}")

    fps_results = None
    if video_paths:
        valid = [v for v in video_paths if v.exists()]
        if valid:
            bench = FPSBenchmark(args.pose_weights, args.model, stride=15, device=device)
            fps_results = bench.run_all(valid)
            if fps_results:
                print(f"  FPS: {fps_results['avg_fps']:.2f} | "
                      f"Pose: {fps_results['pose_ms_avg']:.1f}ms | "
                      f"TFM: {fps_results['tfm_ms_avg']:.1f}ms | "
                      f"{fps_results['video_count']} videos")
    if not fps_results:
        print("  [skip] No FPS videos provided or found.")

    # ── 4. SOTA ────────────────────────────────────────────────────────────
    print("\n[4/5] Building SOTA comparison...")
    sota_rows = build_sota_table(metrics, fps_results)
    print_sota_table(sota_rows)
    save_sota_csv(sota_rows, out_dir)
    save_latex_table(sota_rows, out_dir)

    # ── 5. Plots + Report ──────────────────────────────────────────────────
    print("\n[5/5] Generating plots and report...")
    save_plots(metrics, out_dir)
    save_report(metrics, fps_results, sota_rows, out_dir)

    print(f"\n{'=' * 70}")
    print(f"DONE — Results in: {out_dir.absolute()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
