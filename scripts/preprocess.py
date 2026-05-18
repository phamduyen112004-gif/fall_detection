# Fall Detection - All-In-One Dataset Preprocessing Pipeline
# ============================================================
# Datasets: CaucaFall, MCFD
# Output: Standardized (60, 60) PIFR feature vectors
# ============================================================
# DEPRECATED: Use `python main.py --mode preprocess` instead.
# This script is kept for backward compatibility only.
# ============================================================

import os
import gc
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from src.config import (
    CAUCAFALL_DIR, MCFD_DIR, MCFD_CSV,
    OUTPUT_DIR, YOLO_MODEL, CONF_THRESHOLD,
    TARGET_FRAMES, MAX_FRAMES
)
from src.pifr_features import COCO_IDX, compute_pifr


# COCO Keypoint Indices
COCO_IDX = {
    'nose': 0, 'left_shoulder': 5, 'right_shoulder': 6,
    'left_hip': 11, 'right_hip': 12, 'left_knee': 13,
    'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}


# ============================================================
# PIFR Feature Extraction
# ============================================================

def extract_keypoints(frame, model, w, h):
    """Extract normalized keypoints from frame using YOLO Pose."""
    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    
    if results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None
    
    kpts = results[0].keypoints.data[0].cpu().numpy()
    
    if len(kpts) < 17:
        return None
    
    normalized = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        normalized[i, 0] = kpts[i, 0] / w
        normalized[i, 1] = kpts[i, 1] / h
        normalized[i, 2] = kpts[i, 2]
    
    return normalized


def compute_geometric_features(kpts):
    """Compute 9 geometric features from 17 keypoints."""
    features = []
    x, y = kpts[:, 0], kpts[:, 1]
    
    features.append(np.mean(x))  # F1: CoM X
    features.append(np.mean(y))  # F2: CoM Y
    
    # F3: Shoulder-Nose Angle
    nose = kpts[COCO_IDX['nose']][:2]
    l_sh = kpts[COCO_IDX['left_shoulder']][:2]
    r_sh = kpts[COCO_IDX['right_shoulder']][:2]
    BA, BC = l_sh - nose, r_sh - nose
    nBA, nBC = np.linalg.norm(BA), np.linalg.norm(BC)
    features.append(np.arccos(np.clip(np.dot(BA, BC) / (nBA * nBC + 1e-8), -1, 1)) if nBA > 0 and nBC > 0 else 0.0)
    
    # F4: Torso Angle
    l_hip = kpts[COCO_IDX['left_hip']][:2]
    r_hip = kpts[COCO_IDX['right_hip']][:2]
    mid_hip = (l_hip + r_hip) / 2
    v_torso = mid_hip - nose
    n_vt = np.linalg.norm(v_torso)
    features.append(np.arccos(v_torso[1] / n_vt) if n_vt > 0 else 0.0)
    
    # F5: Hip Angle
    v_hip = r_hip - l_hip
    n_vh = np.linalg.norm(v_hip)
    features.append(np.arccos(v_hip[0] / n_vh) if n_vh > 0 else 0.0)
    
    # F6: Shoulder Angle
    v_sh = r_sh - l_sh
    n_vs = np.linalg.norm(v_sh)
    features.append(np.arccos(v_sh[0] / n_vs) if n_vs > 0 else 0.0)
    
    # F7: Left Leg Angle
    l_knee = kpts[COCO_IDX['left_knee']][:2]
    l_ankle = kpts[COCO_IDX['left_ankle']][:2]
    v1, v2 = l_knee - l_hip, l_ankle - l_knee
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    features.append(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2 + 1e-8), -1, 1)) if n1 > 0 and n2 > 0 else 0.0)
    
    # F8: Right Leg Angle
    r_knee = kpts[COCO_IDX['right_knee']][:2]
    r_ankle = kpts[COCO_IDX['right_ankle']][:2]
    v1, v2 = r_knee - r_hip, r_ankle - r_knee
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    features.append(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2 + 1e-8), -1, 1)) if n1 > 0 and n2 > 0 else 0.0)
    
    # F9: Nose-to-Ankle Angle
    mid_ankle = (l_ankle + r_ankle) / 2
    v_na = mid_ankle - nose
    n_na = np.linalg.norm(v_na)
    features.append(np.arccos(v_na[1] / n_na) if n_na > 0 else 0.0)
    
    return np.array(features, dtype=np.float32)


def extract_pifr(kpts):
    """Extract 60D PIFR feature vector: 51 keypoint values + 9 geometric angles."""
    return np.concatenate([kpts.flatten(), compute_geometric_features(kpts)])


# ============================================================
# Video Processing
# ============================================================

def process_video(video_path, model, fallback):
    """Process full video and extract features."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    features, prev = [], fallback
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        kpts = extract_keypoints(frame, model, w, h)
        
        if kpts is None:
            features.append(prev.copy() if prev is not None else np.zeros(60, dtype=np.float32))
        else:
            vec = extract_pifr(kpts)
            features.append(vec)
            prev = vec
    
    cap.release()
    return np.array(features, dtype=np.float32) if features else None


def process_segment(video_path, start, end, model, fallback):
    """Process video segment and extract features."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    features, prev, cur = [], fallback, 0
    
    while True:
        ret, frame = cap.read()
        if not ret or cur > end:
            break
        
        if start <= cur <= end:
            h, w = frame.shape[:2]
            kpts = extract_keypoints(frame, model, w, h)
            
            if kpts is None:
                features.append(prev.copy() if prev is not None else np.zeros(60, dtype=np.float32))
            else:
                vec = extract_pifr(kpts)
                features.append(vec)
                prev = vec
        
        cur += 1
    
    cap.release()
    return np.array(features, dtype=np.float32) if features else None


def standardize(features):
    """Standardize features to exact (60, 60) shape."""
    from src.utils import standardize_temporal_dim
    return standardize_temporal_dim(features, TARGET_FRAMES, MAX_FRAMES)


def safe_name(s):
    """Sanitize filename."""
    return s.replace(".", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")


def is_fall(name):
    """Check if action is a fall."""
    return "fall" in name.lower()


# ============================================================
# Dataset Processors
# ============================================================

def process_caucafall():
    """Process CaucaFall dataset."""
    print("\n" + "="*50 + "\nProcessing: CaucaFall\n" + "="*50)
    items = []
    
    for subj_dir in sorted(os.listdir(CAUCAFALL_DIR)):
        subj_path = os.path.join(CAUCAFALL_DIR, subj_dir)
        if not os.path.isdir(subj_path) or not subj_dir.startswith("Subject."):
            continue
        
        for action_dir in sorted(os.listdir(subj_path)):
            action_path = os.path.join(subj_path, action_dir)
            if not os.path.isdir(action_path):
                continue
            
            for video in os.listdir(action_path):
                if not video.endswith('.avi'):
                    continue
                
                items.append({
                    'type': 'caucafall',
                    'path': os.path.join(action_path, video),
                    'label': 1 if is_fall(action_dir) else 0,
                    'name': f"caucafall_{safe_name(subj_dir)}_{safe_name(action_dir)}",
                    'subj': subj_dir, 'action': action_dir
                })
    
    return items


def process_mcfd():
    """Process MCFD dataset."""
    print("\n" + "="*50 + "\nProcessing: MCFD\n" + "="*50)
    items = []
    df = pd.read_csv(MCFD_CSV)
    
    for idx, row in df.iterrows():
        chute, cam = int(row['chute']), int(row['cam'])
        start, end = int(row['start']), int(row['end'])
        label = int(row['label'])
        
        video_path = os.path.join(MCFD_DIR, f"chute{chute:02d}", f"cam{cam}.avi")
        
        if os.path.exists(video_path):
            items.append({
                'type': 'mcfd',
                'path': video_path,
                'label': label,
                'name': f"mcfd_chute{chute:02d}_cam{cam}_row{idx}",
                'start': start, 'end': end
            })
    
    return items


def process_vidfall29():
    """Process VidFall-29 dataset. Customize based on actual structure."""
    print("\n" + "="*50 + "\nProcessing: VidFall-29\n" + "="*50)
    items = []
    
    if not os.path.exists(VIDFALL29_DIR):
        print(f"Warning: VidFall-29 not found at {VIDFALL29_DIR}")
        return items
    
    for cat_dir in sorted(os.listdir(VIDFALL29_DIR)):
        cat_path = os.path.join(VIDFALL29_DIR, cat_dir)
        if not os.path.isdir(cat_path):
            continue
        
        for video in sorted(os.listdir(cat_path)):
            if not (video.endswith('.avi') or video.endswith('.mp4')):
                continue
            
            items.append({
                'type': 'vidfall29',
                'path': os.path.join(cat_path, video),
                'label': 1 if is_fall(cat_dir) else 0,
                'name': f"vidfall29_{safe_name(cat_dir)}_{safe_name(video)}",
                'category': cat_dir
            })
    
    return items


# ============================================================
# Main Pipeline
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("AIO FALL DETECTION PREPROCESSING")
    print("Datasets: CaucaFall + MCFD + VidFall-29")
    print(f"Output Shape: ({TARGET_FRAMES}, 60)")
    print("=" * 60)
    
    # Load YOLO once
    model = YOLO(YOLO_MODEL)
    fallback = np.zeros(60, dtype=np.float32)
    
    # Collect all items
    all_items = []
    all_items.extend(process_caucafall())
    all_items.extend(process_mcfd())
    all_items.extend(process_vidfall29())
    
    print(f"\nTotal items: {len(all_items)}\n")
    
    # Process
    processed = skipped = errors = 0
    
    for item in tqdm(all_items, desc="Processing"):
        x_path = os.path.join(OUTPUT_DIR, f"X_{item['name']}.npy")
        y_path = os.path.join(OUTPUT_DIR, f"y_{item['name']}.npy")
        
        if os.path.exists(x_path) and os.path.exists(y_path):
            skipped += 1
            continue
        
        try:
            if item['type'] in ('caucafall', 'vidfall29'):
                features = process_video(item['path'], model, fallback)
            else:
                features = process_segment(item['path'], item['start'], item['end'], model, fallback)
            
            if features is None or len(features) == 0:
                errors += 1
                continue
            
            features = standardize(features)
            np.save(x_path, features)
            np.save(y_path, np.array([item['label']], dtype=np.int32))
            processed += 1
            
        except Exception as e:
            print(f"Error: {item['name']}: {e}")
            errors += 1
        finally:
            if 'features' in locals():
                del features
            gc.collect()
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Output:    {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
