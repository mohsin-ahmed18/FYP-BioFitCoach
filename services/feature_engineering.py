"""
services/feature_engineering.py
=================================
Converts raw MediaPipe pose landmarks into the 48-feature vector
that the BiLSTM model was trained on.

Must match the training notebook EXACTLY:
    Group 1 — raw (x, y, z) for 13 joints  = 39 features
    Group 2 — 9 computed joint angles       =  9 features
    Total                                   = 48 features

If you change anything here, retrain the model.
"""

import numpy as np
import math
from typing import Dict, List, Optional


# ── 13 key joints used during training ──────────────────────────────────────
# Order matters — must match the training notebook
KEY_LANDMARKS: Dict[str, int] = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "nose":            0,
}

NUM_FEATURES    = 48   # 13 × 3 + 9
SEQUENCE_LENGTH = 30   # must match training (set in config.py too)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    """Angle at joint B (degrees) between rays B→A and B→C."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# ── Main extraction ──────────────────────────────────────────────────────────

def extract_features(landmarks) -> Optional[np.ndarray]:
    """
    Extract the 48-feature vector from one MediaPipe result.

    Parameters
    ----------
    landmarks : results.pose_landmarks  (NormalizedLandmarkList)

    Returns
    -------
    np.ndarray of shape (48,) or None if any key joint has visibility < 0.5
    """
    lm = landmarks.landmark

    # Reject frames where any key joint is occluded
    for name, idx in KEY_LANDMARKS.items():
        if lm[idx].visibility < 0.5:
            return None

    # Helper: 2-D point for angle calculation
    def pt(name):
        idx = KEY_LANDMARKS[name]
        return [lm[idx].x, lm[idx].y]

    # ── Group 1: raw x, y, z for 13 joints (39 values) ─────────────────────
    coords: List[float] = []
    for name in KEY_LANDMARKS:          # insertion order (Python 3.7+)
        idx = KEY_LANDMARKS[name]
        coords.extend([lm[idx].x, lm[idx].y, lm[idx].z])

    # ── Group 2: 9 joint angles (9 values) ──────────────────────────────────
    angles: List[float] = [
        _angle(pt("left_shoulder"),  pt("left_elbow"),    pt("left_wrist")),      # left elbow
        _angle(pt("right_shoulder"), pt("right_elbow"),   pt("right_wrist")),     # right elbow
        _angle(pt("left_hip"),       pt("left_knee"),     pt("left_ankle")),      # left knee
        _angle(pt("right_hip"),      pt("right_knee"),    pt("right_ankle")),     # right knee
        _angle(pt("left_shoulder"),  pt("left_hip"),      pt("left_knee")),       # left hip
        _angle(pt("right_shoulder"), pt("right_hip"),     pt("right_knee")),      # right hip
        _angle(pt("left_elbow"),     pt("left_shoulder"), pt("right_shoulder")),  # shoulder spread
        _angle(pt("left_shoulder"),  pt("left_hip"),      pt("left_knee")),       # torso
        _angle(pt("left_knee"),      pt("left_hip"),      pt("right_hip")),       # hip width
    ]

    return np.array(coords + angles, dtype=np.float32)


def extract_rule_angles(landmarks) -> Dict[str, float]:
    """
    Extended angle set used ONLY by form_rules.py — not fed to the model.
    Returns an empty dict if landmarks are missing.
    """
    try:
        lm = landmarks.landmark

        def pt(name):
            idx = KEY_LANDMARKS[name]
            return [lm[idx].x, lm[idx].y]

        return {
            "left_elbow_angle":     _angle(pt("left_shoulder"),  pt("left_elbow"),   pt("left_wrist")),
            "right_elbow_angle":    _angle(pt("right_shoulder"), pt("right_elbow"),  pt("right_wrist")),
            "left_knee_angle":      _angle(pt("left_hip"),       pt("left_knee"),    pt("left_ankle")),
            "right_knee_angle":     _angle(pt("right_hip"),      pt("right_knee"),   pt("right_ankle")),
            "left_hip_angle":       _angle(pt("left_shoulder"),  pt("left_hip"),     pt("left_knee")),
            "right_hip_angle":      _angle(pt("right_shoulder"), pt("right_hip"),    pt("right_knee")),
            "left_shoulder_angle":  _angle(pt("left_hip"),       pt("left_shoulder"),pt("left_elbow")),
            "right_shoulder_angle": _angle(pt("right_hip"),      pt("right_shoulder"),pt("right_elbow")),
            "torso_angle":          _angle(pt("left_shoulder"),  pt("left_hip"),     pt("left_knee")),
        }
    except Exception:
        return {}