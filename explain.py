#!/usr/bin/env python3
"""
Explainability Module for Fall Detection System (XAI).

Tác giả: Fall Detection Team
Ngày: May 2026

Mô tả:
    Script sinh qualitative results cho thesis, giải thích cách model
    đưa ra quyết định fall detection.

Phương pháp:
    1. Gradient-based Feature Importance (Backpropagation)
    2. Attention Weight Analysis (Transformer)
    3. Integrated Gradients (XAI)
    4. Permutation Feature Importance

Visualizations:
    - Keypoint Importance Heatmap (color-coded skeleton)
    - Feature Attribution Bar Chart
    - Temporal Attention Heatmap
    - Grad-CAM style overlay

Sử dụng:
    python explain.py --model best_model.pth --input data/sample_sequence.npy --output explain_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Project root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60
FEATURE_DIM = 60

# COCO Keypoint connections for skeleton
COCO_SKELETON = [
    (0, 1),   # nose-left_eye
    (0, 2),   # nose-right_eye
    (1, 3),   # left_eye-left_ear
    (2, 4),   # right_eye-right_ear
    (5, 6),   # left_shoulder-right_shoulder
    (5, 7),   # left_shoulder-left_elbow
    (7, 9),   # left_elbow-left_wrist
    (6, 8),   # right_shoulder-right_elbow
    (8, 10),  # right_elbow-right_wrist
    (5, 11),  # left_shoulder-left_hip
    (6, 12),  # right_shoulder-right_hip
    (11, 12), # left_hip-right_hip
    (11, 13), # left_hip-left_knee
    (13, 15), # left_knee-left_ankle
    (12, 14), # right_hip-right_knee
    (14, 16), # right_knee-right_ankle
]

COCO_KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

GEOMETRIC_FEATURE_NAMES = [
    "center_mass_x", "center_mass_y",
    "shoulder_nose_angle", "torso_angle",
    "hip_angle", "shoulder_angle",
    "left_leg_angle", "right_leg_angle",
    "nose_to_ankle_angle",
]

# Colormaps for importance visualization
IMPORTANCE_CMAP = LinearSegmentedColormap.from_list(
    "importance", ["#FFFF00", "#FFA500", "#FF0000", "#8B0000"]
)  # Yellow -> Orange -> Red -> Dark Red


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient-Based Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

class GradientFeatureImportance:
    """
    Tính feature importance bằng gradient backpropagation.

    Phương pháp:
        1. Forward pass input qua model
        2. Tính gradient của output w.r.t. input: d(output)/d(input)
        3. Importance = |gradient| hoặc gradient * input

    Mathematical Explanation:
        ∂L/∂x_i = ∂L/∂y * ∂y/∂x_i
        Importance_i = |∂L/∂x_i| * |x_i|
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register on the final layer
        if hasattr(self.model, "fc"):
            self.model.fc.register_full_backward_hook(backward_hook)

    def compute(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Tính feature importance scores.

        Args:
            x: Input tensor shape (1, seq_len, feature_dim)
            target_class: Target class index (1 = fall)

        Returns:
            Importance scores shape (seq_len, feature_dim)
        """
        x = x.clone().requires_grad_(True)
        self.model.eval()

        # Forward pass
        output = self.model(x)

        # Get probability for target class
        if target_class == 1:
            loss = output.squeeze()
        else:
            loss = 1 - output.squeeze()

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get gradients
        if x.grad is not None:
            gradients = x.grad.data.cpu().numpy()
        else:
            gradients = np.zeros_like(x.data.cpu().numpy())

        # Compute importance: |gradient| * |input|
        importance = np.abs(gradients) * np.abs(x.data.cpu().numpy())

        # Normalize to [0, 1]
        importance = importance.squeeze()
        if importance.max() > 0:
            importance = importance / importance.max()

        return importance

    def get_keypoint_importance(self, importance: np.ndarray) -> np.ndarray:
        """
        Extract keypoint importance from full feature importance.

        Args:
            importance: Shape (seq_len, 60)

        Returns:
            Shape (17,) - average importance across sequence for each keypoint
        """
        # First 51 dims are 17 keypoints * 3 (x, y, conf)
        keypoint_importance = np.zeros(17)
        for i in range(17):
            keypoint_importance[i] = np.mean(importance[:, i*3:(i+1)*3])

        return keypoint_importance

    def get_geometric_importance(self, importance: np.ndarray) -> np.ndarray:
        """
        Extract geometric feature importance.

        Args:
            importance: Shape (seq_len, 60)

        Returns:
            Shape (9,) - average importance for each geometric feature
        """
        # Last 9 dims are geometric features
        geometric_importance = np.zeros(9)
        for i in range(9):
            geometric_importance[i] = np.mean(importance[:, 51 + i])

        return geometric_importance


# ═══════════════════════════════════════════════════════════════════════════════
# Integrated Gradients
# ═══════════════════════════════════════════════════════════════════════════════

class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al., 2017) cho XAI.

    Mathematical Definition:
        IG_i(x) = (x_i - x'_i) * ∫_0^1 ∂F(x' + α(x - x'_i))/∂x_i dα

    trong đó:
        - F() là model output
        - x là input gốc
        - x' là baseline (thường là zero)
        - α là interpolation factor
    """

    def __init__(self, model: nn.Module, baseline: torch.Tensor | None = None):
        self.model = model
        self.baseline = baseline if baseline is not None else torch.zeros(1, SEQ_LEN, FEATURE_DIM)

    def compute(
        self,
        x: torch.Tensor,
        steps: int = 50,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Tính integrated gradients.

        Args:
            x: Input tensor
            steps: Số bước interpolation
            target_class: Target class

        Returns:
            Integrated gradients shape giống input
        """
        x = x.clone()
        baseline = self.baseline.clone().to(x.device)

        if baseline.shape != x.shape:
            baseline = torch.zeros_like(x)

        # Interpolate between baseline and input
        alphas = np.linspace(0, 1, steps)
        gradients = []

        for alpha in tqdm(alphas, desc="Computing Integrated Gradients"):
            interpolated = (1 - alpha) * baseline + alpha * x
            interpolated = interpolated.clone().requires_grad_(True)

            output = self.model(interpolated)

            if target_class == 1:
                loss = output.squeeze()
            else:
                loss = 1 - output.squeeze()

            self.model.zero_grad()
            loss.backward()

            if interpolated.grad is not None:
                gradients.append(interpolated.grad.data.cpu().numpy())

        # Average gradients
        avg_gradients = np.mean(gradients, axis=0)

        # Compute integrated gradients
        integrated_grads = (x.data.cpu().numpy() - baseline.cpu().numpy()) * avg_gradients

        # Normalize
        integrated_grads = np.abs(integrated_grads).squeeze()
        if integrated_grads.max() > 0:
            integrated_grads = integrated_grads / integrated_grads.max()

        return integrated_grads


# ═══════════════════════════════════════════════════════════════════════════════
# Attention Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionAnalyzer:
    """
    Phân tích attention weights từ Transformer.

    Trích xuất attention patterns để hiểu:
        1. Temporal attention: Frame nào được attend nhiều nhất
        2. Feature attention: Feature nào được attend nhiều nhất
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = []
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # Attention weights are in output[1] for MultiheadAttention
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_weights.append(output[1].detach())
            return hook

        # Find transformer encoder layers
        if hasattr(self.model, "transformer_encoder"):
            for i, layer in enumerate(self.model.transformer_encoder.layers):
                if hasattr(layer, "self_attn"):
                    layer.self_attn.register_forward_hook(get_attention_hook(i))

    def reset(self):
        """Reset attention weights buffer."""
        self.attention_weights = []

    def get_temporal_attention(self) -> np.ndarray:
        """
        Get average attention across all heads and layers.

        Returns:
            Shape (seq_len, seq_len) - attention matrix
        """
        if not self.attention_weights:
            return np.zeros((SEQ_LEN, SEQ_LEN))

        # Average across layers and heads
        attn = torch.stack(self.attention_weights, dim=0)  # (n_layers, batch, heads, seq, seq)
        attn = attn.mean(dim=(0, 2))  # (batch, seq, seq)
        return attn[0].cpu().numpy()

    def get_frame_importance(self) -> np.ndarray:
        """
        Get importance of each frame based on attention.

        Returns:
            Shape (seq_len,) - importance score per frame
        """
        attn_matrix = self.get_temporal_attention()

        # Sum attention received by each frame
        frame_importance = attn_matrix.sum(axis=0)
        frame_importance = frame_importance / frame_importance.max() if frame_importance.max() > 0 else frame_importance

        return frame_importance


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════════

def draw_skeleton_with_importance(
    keypoints: np.ndarray,
    importance: np.ndarray,
    image: np.ndarray | None = None,
    skeleton: list = COCO_SKELETON,
    keypoint_names: list = COCO_KEYPOINT_NAMES,
    figsize: tuple = (12, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Vẽ skeleton với color-coded importance.

    Args:
        keypoints: Keypoints shape (17, 3) [x, y, conf]
        importance: Importance scores shape (17,) normalized [0, 1]
        image: Optional background image
        skeleton: List of connections
        keypoint_names: Names for legend
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw background if provided
    if image is not None:
        ax.imshow(image)
    else:
        # Create blank image
        h, w = 480, 640
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.invert_yaxis()
        ax.set_facecolor('#f0f0f0')

    # Draw skeleton connections
    for (i, j) in skeleton:
        if importance[i] > 0.1 or importance[j] > 0.1:
            x1, y1 = keypoints[i, 0] * w, keypoints[i, 1] * h
            x2, y2 = keypoints[j, 0] * w, keypoints[j, 1] * h

            # Average importance for the bone
            bone_importance = (importance[i] + importance[j]) / 2

            # Color based on importance
            color = IMPORTANCE_CMAP(bone_importance)
            linewidth = 2 + bone_importance * 4

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.9, zorder=1)

    # Draw keypoints
    for i, (kp, imp) in enumerate(zip(keypoints, importance)):
        x, y = kp[0] * w, kp[1] * h
        conf = kp[2]

        if conf < 0.2:
            continue

        color = IMPORTANCE_CMAP(imp)
        size = 100 + imp * 200

        ax.scatter(x, y, c=[color], s=size, edgecolors='white', linewidths=1.5, zorder=2)

        # Add keypoint label for high importance points
        if imp > 0.5:
            ax.annotate(
                keypoint_names[i],
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
            )

    # Create colorbar legend
    sm = plt.cm.ScalarMappable(cmap=IMPORTANCE_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Importance Score', fontsize=12)
    cbar.ax.set_ylabel('Low (Yellow) → High (Red)', fontsize=10)

    # Title
    ax.set_title('Keypoint Importance Heatmap\n(Color: Yellow=Low → Dark Red=High)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"  [Saved] {save_path}")

    return fig


def plot_feature_importance_bar(
    keypoint_importance: np.ndarray,
    geometric_importance: np.ndarray,
    top_k: int = 10,
    figsize: tuple = (14, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Vẽ bar chart showing top-k most important features.

    Args:
        keypoint_importance: Shape (17,)
        geometric_importance: Shape (9,)
        top_k: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Combine all features
    all_names = COCO_KEYPOINT_NAMES + GEOMETRIC_FEATURE_NAMES
    all_importance = np.concatenate([keypoint_importance, geometric_importance])

    # Sort by importance
    sorted_indices = np.argsort(all_importance)[::-1]
    top_indices = sorted_indices[:top_k]

    top_names = [all_names[i] for i in top_indices]
    top_values = [all_importance[i] for i in top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars based on importance
    colors = [IMPORTANCE_CMAP(v) for v in top_values]

    bars = ax.barh(range(len(top_names)), top_values, color=colors, edgecolor='black', linewidth=0.5)

    # Customize
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=11)
    ax.set_xlabel('Importance Score (Normalized)', fontsize=12)
    ax.set_title(f'Top {top_k} Most Important Features for Fall Detection', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_values)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=IMPORTANCE_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Importance', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"  [Saved] {save_path}")

    return fig


def plot_temporal_attention_heatmap(
    attention_matrix: np.ndarray,
    frame_importance: np.ndarray,
    figsize: tuple = (16, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Vẽ temporal attention analysis.

    Args:
        attention_matrix: Shape (seq_len, seq_len)
        frame_importance: Shape (seq_len,)
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [2, 1, 2]})

    # 1. Attention Matrix Heatmap
    ax1 = axes[0]
    sns.heatmap(
        attention_matrix,
        cmap='YlOrRd',
        ax=ax1,
        cbar_kws={'label': 'Attention Weight'},
        xticklabels=10,
        yticklabels=10,
    )
    ax1.set_xlabel('Key Frame Index', fontsize=11)
    ax1.set_ylabel('Query Frame Index', fontsize=11)
    ax1.set_title('Transformer Attention Matrix', fontsize=12, fontweight='bold')

    # 2. Frame Importance
    ax2 = axes[1]
    colors = [IMPORTANCE_CMAP(v) for v in frame_importance]
    ax2.barh(range(len(frame_importance)), frame_importance, color=colors)
    ax2.set_xlabel('Importance', fontsize=11)
    ax2.set_ylabel('Frame Index', fontsize=11)
    ax2.set_title('Frame Importance', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # 3. Feature Importance Over Time (aggregated)
    ax3 = axes[2]

    # Show attention for specific important frames
    important_frames = np.argsort(frame_importance)[-5:][::-1]
    for f_idx in important_frames[:3]:
        ax3.plot(attention_matrix[f_idx], label=f'Frame {f_idx}', linewidth=1.5, alpha=0.8)

    ax3.set_xlabel('Attended Frame', fontsize=11)
    ax3.set_ylabel('Attention Weight', fontsize=11)
    ax3.set_title('Attention from Top-3 Important Frames', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)

    plt.suptitle('Temporal Attention Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"  [Saved] {save_path}")

    return fig


def plot_integrated_gradients(
    ig_importance: np.ndarray,
    keypoint_importance: np.ndarray,
    geometric_importance: np.ndarray,
    figsize: tuple = (16, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Vẽ Integrated Gradients visualization.

    Args:
        ig_importance: Full importance from IG (seq_len, 60)
        keypoint_importance: Aggregated keypoint importance
        geometric_importance: Aggregated geometric importance
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Feature Importance Timeline
    ax1 = axes[0]
    # Sum importance over features for each frame
    frame_importance = ig_importance.mean(axis=1)
    ax1.plot(frame_importance, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.fill_between(range(len(frame_importance)), frame_importance, alpha=0.3)
    ax1.set_xlabel('Frame Index (Sequence of 60)', fontsize=11)
    ax1.set_ylabel('Integrated Gradient Magnitude', fontsize=11)
    ax1.set_title('Feature Importance Over Time', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Highlight high importance frames
    threshold = np.mean(frame_importance) + np.std(frame_importance)
    high_importance_frames = np.where(frame_importance > threshold)[0]
    if len(high_importance_frames) > 0:
        ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (μ+σ={threshold:.3f})')
        for f in high_importance_frames[:10]:  # Limit annotations
            ax1.annotate(f'F{f}', (f, frame_importance[f]), fontsize=8, color='red')

    ax1.legend()

    # 2. Keypoint Importance Pie Chart
    ax2 = axes[1]
    top_k = 5
    top_indices = np.argsort(keypoint_importance)[::-1][:top_k]
    top_names = [COCO_KEYPOINT_NAMES[i] for i in top_indices]
    top_values = [keypoint_importance[i] for i in top_indices]

    colors = [IMPORTANCE_CMAP(v / keypoint_importance.max()) for v in top_values]
    wedges, texts, autotexts = ax2.pie(
        top_values,
        labels=top_names,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05] * len(top_names),
        shadow=True,
    )
    ax2.set_title(f'Top {top_k} Keypoint Attribution', fontsize=12, fontweight='bold')

    # 3. Geometric Feature Importance
    ax3 = axes[2]
    sorted_idx = np.argsort(geometric_importance)[::-1]
    colors = [IMPORTANCE_CMAP(geometric_importance[i] / geometric_importance.max()) for i in sorted_idx]

    ax3.barh(
        [GEOMETRIC_FEATURE_NAMES[i] for i in sorted_idx],
        [geometric_importance[i] for i in sorted_idx],
        color=colors,
        edgecolor='black',
    )
    ax3.set_xlabel('Attribution Score', fontsize=11)
    ax3.set_title('Geometric Feature Attribution', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    plt.suptitle('Integrated Gradients Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"  [Saved] {save_path}")

    return fig


def plot_mechanistic_analysis_summary(
    keypoint_importance: np.ndarray,
    geometric_importance: np.ndarray,
    attention_matrix: np.ndarray,
    prediction: float,
    label: int,
    figsize: tuple = (20, 12),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Tạo comprehensive mechanistic analysis figure cho thesis.

    Args:
        keypoint_importance: Shape (17,)
        geometric_importance: Shape (9,)
        attention_matrix: Shape (60, 60)
        prediction: Model prediction probability
        label: True label (0=nofall, 1=fall)
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Title with prediction info
    label_str = "FALL" if label == 1 else "NO-FALL"
    pred_str = "FALL" if prediction > 0.5 else "NO-FALL"
    fig.suptitle(
        f'Mechanistic Analysis: Ground Truth = {label_str} | Model Prediction = {pred_str} ({prediction:.3f})',
        fontsize=16, fontweight='bold', y=0.98
    )

    # 1. Keypoint Importance (top-left, larger)
    ax1 = fig.add_subplot(gs[0, 0:2])
    sorted_idx = np.argsort(keypoint_importance)[::-1]
    colors = [IMPORTANCE_CMAP(keypoint_importance[i]) for i in sorted_idx]
    ax1.barh(
        [COCO_KEYPOINT_NAMES[i] for i in sorted_idx],
        [keypoint_importance[i] for i in sorted_idx],
        color=colors,
    )
    ax1.set_xlabel('Importance', fontsize=11)
    ax1.set_title('(a) Keypoint Attribution', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    # 2. Geometric Features (top-right)
    ax2 = fig.add_subplot(gs[0, 2:4])
    sorted_idx = np.argsort(geometric_importance)[::-1]
    colors = [IMPORTANCE_CMAP(geometric_importance[i]) for i in sorted_idx]
    ax2.barh(
        [GEOMETRIC_FEATURE_NAMES[i] for i in sorted_idx],
        [geometric_importance[i] for i in sorted_idx],
        color=colors,
    )
    ax2.set_xlabel('Importance', fontsize=11)
    ax2.set_title('(b) Geometric Feature Attribution', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    # 3. Attention Matrix (middle-left, larger)
    ax3 = fig.add_subplot(gs[1, 0:2])
    sns.heatmap(
        attention_matrix,
        cmap='YlOrRd',
        ax=ax3,
        cbar_kws={'label': 'Attention'},
    )
    ax3.set_xlabel('Frame Index', fontsize=11)
    ax3.set_ylabel('Frame Index', fontsize=11)
    ax3.set_title('(c) Transformer Attention Pattern', fontsize=12, fontweight='bold')

    # 4. Frame Importance (middle-right)
    ax4 = fig.add_subplot(gs[1, 2:4])
    frame_importance = attention_matrix.sum(axis=0)
    frame_importance = frame_importance / frame_importance.max() if frame_importance.max() > 0 else frame_importance
    colors = [IMPORTANCE_CMAP(v) for v in frame_importance]
    ax4.bar(range(len(frame_importance)), frame_importance, color=colors, width=1.0)
    ax4.set_xlabel('Frame Index', fontsize=11)
    ax4.set_ylabel('Attended Importance', fontsize=11)
    ax4.set_title('(d) Per-Frame Attention', fontsize=12, fontweight='bold')

    # 5. Human Pose Diagram with Importance (bottom-left, spanning 2 columns)
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 100)
    ax5.set_aspect('equal')
    ax5.axis('off')
    ax5.set_title('(e) Keypoint Importance Visualization', fontsize=12, fontweight='bold')

    # Draw simplified skeleton
    skeleton_coords = {
        'nose': (50, 85),
        'l_shoulder': (35, 65),
        'r_shoulder': (65, 65),
        'l_hip': (40, 40),
        'r_hip': (60, 40),
        'l_knee': (35, 20),
        'r_knee': (65, 20),
        'l_ankle': (35, 5),
        'r_ankle': (65, 5),
    }

    # Draw connections
    connections = [
        ('nose', 'l_shoulder'), ('nose', 'r_shoulder'),
        ('l_shoulder', 'r_shoulder'),
        ('l_shoulder', 'l_hip'), ('r_shoulder', 'r_hip'),
        ('l_hip', 'r_hip'),
        ('l_hip', 'l_knee'), ('r_hip', 'r_knee'),
        ('l_knee', 'l_ankle'), ('r_knee', 'r_ankle'),
    ]

    for (a, b) in connections:
        ax5.plot([skeleton_coords[a][0], skeleton_coords[b][0]],
                 [skeleton_coords[a][1], skeleton_coords[b][1]],
                 'k-', linewidth=2, alpha=0.5)

    # Draw keypoints with importance color
    kp_importance_map = {name: keypoint_importance[i] for i, name in enumerate(COCO_KEYPOINT_NAMES)}
    for name, (x, y) in skeleton_coords.items():
        imp = kp_importance_map.get(name, 0)
        color = IMPORTANCE_CMAP(imp)
        ax5.scatter(x, y, c=[color], s=300, edgecolors='black', linewidths=1.5, zorder=5)
        ax5.annotate(name.replace('_', '\n'), (x, y), xytext=(5, 5),
                     textcoords='offset points', fontsize=7, ha='left')

    # 6. Summary Statistics (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.axis('off')

    # Top 5 features
    all_names = COCO_KEYPOINT_NAMES + GEOMETRIC_FEATURE_NAMES
    all_importance = np.concatenate([keypoint_importance, geometric_importance])
    top5_idx = np.argsort(all_importance)[::-1][:5]
    top5_info = [(all_names[i], all_importance[i]) for i in top5_idx]

    summary_text = "KEY FINDINGS:\n\n"
    summary_text += "Top 5 Attributed Features:\n"
    for i, (name, imp) in enumerate(top5_info, 1):
        summary_text += f"  {i}. {name}: {imp:.4f}\n"

    summary_text += f"\nAttention Pattern: "
    if attention_matrix.diagonal().mean() > 0.3:
        summary_text += "Strong self-attention\n"
    else:
        summary_text += "Cross-frame attention\n"

    summary_text += f"\nModel Confidence: {prediction:.3f}\n"
    summary_text += f"Classification: {pred_str}"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
              family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"  [Saved] {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main Explainability Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def explain_sample(
    model: nn.Module,
    x: np.ndarray,
    label: int,
    output_dir: str,
    sample_name: str = "sample",
) -> dict[str, Any]:
    """
    Main function to explain a single sample.

    Args:
        model: Trained model
        x: Input features shape (1, seq_len, feature_dim)
        label: Ground truth label
        output_dir: Output directory
        sample_name: Name for saving files

    Returns:
        Dict containing importance scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPLAINING SAMPLE: {sample_name}")
    print(f"{'='*60}")

    # Convert to tensor
    x_tensor = torch.from_numpy(x).float().to(DEVICE)

    # ─── 1. Gradient-based Importance ───
    print("\n[1/4] Computing Gradient-based Feature Importance...")
    grad_analyzer = GradientFeatureImportance(model)
    grad_importance = grad_analyzer.compute(x_tensor, target_class=label)

    keypoint_importance = grad_analyzer.get_keypoint_importance(grad_importance)
    geometric_importance = grad_analyzer.get_geometric_importance(grad_importance)

    print(f"  Keypoint importance shape: {keypoint_importance.shape}")
    print(f"  Top keypoint: {COCO_KEYPOINT_NAMES[np.argmax(keypoint_importance)]} ({keypoint_importance.max():.4f})")
    print(f"  Geometric importance shape: {geometric_importance.shape}")
    print(f"  Top geometric: {GEOMETRIC_FEATURE_NAMES[np.argmax(geometric_importance)]} ({geometric_importance.max():.4f})")

    # ─── 2. Integrated Gradients ───
    print("\n[2/4] Computing Integrated Gradients...")
    ig_analyzer = IntegratedGradients(model)
    ig_importance = ig_analyzer.compute(x_tensor, steps=50, target_class=label)

    ig_keypoint = np.zeros(17)
    for i in range(17):
        ig_keypoint[i] = np.mean(ig_importance[:, i*3:(i+1)*3])

    ig_geometric = np.zeros(9)
    for i in range(9):
        ig_geometric[i] = np.mean(ig_importance[:, 51 + i])

    print(f"  IG computed for {ig_importance.shape[0]} frames")

    # ─── 3. Attention Analysis ───
    print("\n[3/4] Analyzing Transformer Attention...")
    attn_analyzer = AttentionAnalyzer(model)
    attn_analyzer.reset()

    # Forward pass to capture attention
    with torch.no_grad():
        _ = model(x_tensor)

    attention_matrix = attn_analyzer.get_temporal_attention()
    frame_importance = attn_analyzer.get_frame_importance()

    print(f"  Attention matrix shape: {attention_matrix.shape}")
    print(f"  Most attended frame: {np.argmax(frame_importance)}")

    # ─── 4. Model Prediction ───
    print("\n[4/4] Getting Model Prediction...")
    model.eval()
    with torch.no_grad():
        output = model(x_tensor)
        prediction = torch.sigmoid(output).item()

    print(f"  Prediction: {prediction:.4f} ({'FALL' if prediction > 0.5 else 'NO-FALL'})")
    print(f"  Ground Truth: {label} ({'FALL' if label == 1 else 'NO-FALL'})")

    # ─── 5. Generate Visualizations ───
    print("\n[GENERATING] Creating visualizations...")

    viz_dir = output_dir / sample_name
    viz_dir.mkdir(exist_ok=True)

    # Figure 1: Feature Importance Bar Chart
    plot_feature_importance_bar(
        keypoint_importance,
        geometric_importance,
        top_k=10,
        save_path=str(viz_dir / "feature_importance_bar.png"),
    )

    # Figure 2: Temporal Attention Heatmap
    plot_temporal_attention_heatmap(
        attention_matrix,
        frame_importance,
        save_path=str(viz_dir / "temporal_attention.png"),
    )

    # Figure 3: Integrated Gradients
    plot_integrated_gradients(
        ig_importance,
        ig_keypoint,
        ig_geometric,
        save_path=str(viz_dir / "integrated_gradients.png"),
    )

    # Figure 4: Comprehensive Mechanistic Analysis (for thesis)
    plot_mechanistic_analysis_summary(
        keypoint_importance,
        geometric_importance,
        attention_matrix,
        prediction,
        label,
        save_path=str(viz_dir / "mechanistic_analysis.png"),
    )

    # ─── 6. Save Results ───
    results = {
        "sample_name": sample_name,
        "prediction": float(prediction),
        "ground_truth": int(label),
        "keypoint_importance": keypoint_importance.tolist(),
        "geometric_importance": geometric_importance.tolist(),
        "ig_keypoint_importance": ig_keypoint.tolist(),
        "ig_geometric_importance": ig_geometric.tolist(),
        "attention_matrix": attention_matrix.tolist(),
        "frame_importance": frame_importance.tolist(),
        "top_5_features": [
            {"name": COCO_KEYPOINT_NAMES[i] if i < 17 else GEOMETRIC_FEATURE_NAMES[i - 17],
             "importance": float(np.concatenate([keypoint_importance, geometric_importance])[i])}
            for i in np.argsort(np.concatenate([keypoint_importance, geometric_importance]))[::-1][:5]
        ],
    }

    # Save JSON
    json_path = viz_dir / "importance_scores.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [Saved] {json_path}")

    print(f"\n[COMPLETE] Visualizations saved to: {viz_dir}")
    return results


def create_comparison_visualization(
    results_list: list[dict],
    output_path: str,
) -> plt.Figure:
    """
    Tạo comparison visualization cho multiple samples.

    Args:
        results_list: List of results from explain_sample
        output_path: Path to save

    Returns:
        matplotlib Figure
    """
    n_samples = len(results_list)
    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 10))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, results in enumerate(results_list):
        # Keypoint importance comparison
        ax1 = axes[0, i]
        kp_imp = np.array(results["keypoint_importance"])
        colors = [IMPORTANCE_CMAP(v / kp_imp.max()) for v in kp_imp]

        ax1.barh(COCO_KEYPOINT_NAMES, kp_imp, color=colors)
        ax1.set_xlabel('Importance')
        ax1.set_title(f'{results["sample_name"]}\nGT={results["ground_truth"]}, Pred={results["prediction"]:.2f}')
        ax1.invert_yaxis()

        # Geometric importance comparison
        ax2 = axes[1, i]
        geo_imp = np.array(results["geometric_importance"])
        colors = [IMPORTANCE_CMAP(v / geo_imp.max()) for v in geo_imp]

        ax2.barh(GEOMETRIC_FEATURE_NAMES, geo_imp, color=colors)
        ax2.set_xlabel('Importance')
        ax2.set_title('Geometric Features')
        ax2.invert_yaxis()

    plt.suptitle('Feature Importance Comparison Across Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [Saved] {output_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Explainability Module for Fall Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain a single sample
  python explain.py --model best_model.pth --input data/test_sample.npy --label 1

  # Explain batch of samples
  python explain.py --model best_model.pth --input data/test_batch.npy --labels 0,1,1,0

  # Use pre-computed features
  python explain.py --model best_model.pth --features-dir data/processed --output explain_results
        """,
    )
    parser.add_argument("--model", type=str, default="best_hybrid_transformer.pth", help="Model path")
    parser.add_argument("--input", type=str, help="Input numpy file")
    parser.add_argument("--features-dir", type=str, help="Features directory")
    parser.add_argument("--labels", type=str, help="Comma-separated labels")
    parser.add_argument("--output", type=str, default="explain_results", help="Output directory")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)

    args = parser.parse_args()

    # Device
    if args.device:
        DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}")

    # ─── Load Model ───
    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=DEVICE)
    model = HybridFallTransformer(seq_len=SEQ_LEN, feature_dim=FEATURE_DIM)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print("[OK] Model loaded")

    # ─── Load Data ───
    if args.input:
        # Single sample
        x = np.load(args.input)
        if x.ndim == 2:
            x = x[np.newaxis, :, :]  # Add batch dim
        labels = [int(l) for l in (args.labels or "1").split(",")]
        for i, label in enumerate(labels):
            sample_name = f"sample_{i}"
            explain_sample(model, x[i:i+1], label, args.output, sample_name)

    elif args.features_dir:
        # Batch from directory
        import os
        features_dir = Path(args.features_dir)

        # Load test data
        X_path = features_dir / "X_test.npy"
        y_path = features_dir / "y_test.npy"

        if not X_path.exists():
            print(f"[ERROR] Test features not found at {X_path}")
            sys.exit(1)

        X = np.load(X_path)
        y = np.load(y_path).ravel()

        print(f"Loaded {len(X)} test samples")

        # Explain first N samples
        n_explain = min(10, len(X))
        results_list = []

        for i in tqdm(range(n_explain), desc="Explaining samples"):
            results = explain_sample(model, X[i:i+1], int(y[i]), args.output, f"test_sample_{i}")
            results_list.append(results)

        # Create comparison visualization
        create_comparison_visualization(
            results_list,
            str(Path(args.output) / "comparison_all_samples.png"),
        )

    else:
        print("[ERROR] Please specify --input or --features-dir")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("EXPLANATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
