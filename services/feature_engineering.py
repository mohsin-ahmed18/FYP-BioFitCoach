"""
services/feature_engineering.py — Pose Feature Extraction
===========================================================
Converts raw MediaPipe pose landmarks into the 48-feature vector
that the BiLSTM model was trained on.

CRITICAL: This must match your notebook's extract_frame_features() exactly.
    - 13 key joints × 3 coords (x, y, z) = 39 features  [Group 1]
    - 9 joint angles                       =  9 features  [Group 2]
    - Total = 48 features per frame
    - SEQUENCE_LENGTH = 30 frames per BiLSTM input

Vocabulary:
- Landmark      : One detected body joint with x, y, z, visibility.
                  MediaPipe returns 33 landmarks per frame.
- visibility    : Confidence [0–1] that the landmark is visible.
                  We skip frames where any key joint < 0.5.
- Normalized    : x,y coords are in [0,1] relative to frame size,
                  so the model works regardless of camera resolution.
"""

import numpy as np
import math
from typing import Dict, List, Optional


# ── Key landmark indices (subset of MediaPipe's 33) ──────────────────────
# Ordered dict — insertion order is preserved (Python 3.7+)
# This ORDER must match exactly what your notebook used when building features
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

NUM_FEATURES    = 13 * 3 + 9    # = 48
SEQUENCE_LENGTH = 30            # must match notebook SEQUENCE_LENGTH


# ── Geometry helpers ──────────────────────────────────────────────────────

def _angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Angle in degrees at joint B between segments B→A and B→C.
    Uses dot-product formula: cos(θ) = (BA · BC) / (|BA| |BC|)
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# ── Main extraction functions ─────────────────────────────────────────────

def extract_frame_features(landmarks) -> Optional[np.ndarray]:
    """
    Extract the 48-feature vector from one MediaPipe pose result.

    Returns np.ndarray of shape (48,) or None if any key joint
    has low visibility (occluded / out of frame).

    This function is called for EVERY video frame during inference.
    """
    lm = landmarks.landmark

    # Skip frames where any key joint is hidden
    for name, idx in KEY_LANDMARKS.items():
        if lm[idx].visibility < 0.5:
            return None

    def pt(name: str) -> List[float]:
        """Get [x, y] for angle calculation."""
        idx = KEY_LANDMARKS[name]
        return [lm[idx].x, lm[idx].y]

    # ── Group 1: Raw x, y, z coordinates (39 values) ──────────────────
    # Order follows KEY_LANDMARKS insertion order
    coords = []
    for name in KEY_LANDMARKS:
        idx = KEY_LANDMARKS[name]
        coords.extend([lm[idx].x, lm[idx].y, lm[idx].z])

    # ── Group 2: Computed joint angles (9 values) ──────────────────────
    angles = [
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

    return np.array(coords + angles, dtype=np.float32)  # shape: (48,)


def extract_rule_angles(landmarks) -> Dict[str, float]:
    """
    Extended angle set used ONLY by form_rules.py — NOT fed to the BiLSTM.
    These angles make the form rules more readable and biomechanically precise.

    Returns empty dict if any key landmark is missing.
    """
    lm = landmarks.landmark

    for name, idx in KEY_LANDMARKS.items():
        if lm[idx].visibility < 0.3:
            return {}

    def pt(name: str) -> List[float]:
        idx = KEY_LANDMARKS[name]
        return [lm[idx].x, lm[idx].y]

    try:
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
            "hip_width_angle":      _angle(pt("left_knee"),      pt("left_hip"),     pt("right_hip")),
        }
    except Exception:
        return {}