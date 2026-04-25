from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Rep:
    rep_number: int
    quality: str  # "good" | "partial" | "incorrect"
    feedback: str
    tempo: Optional[str] = None
    depth: Optional[str] = None
    stability: Optional[str] = None


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2)


def _smooth(values: List[float], window: int = 5) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]
    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        a = max(0, i - half)
        b = min(len(values), i + half + 1)
        out.append(_median(values[a:b]))
    return out


def _classify_rep(min_angle: float, frames_in_rep: int, fps: float, mode: str) -> Tuple[str, str, str, str, str]:
    """
    mode:
      - "squat": min angle indicates depth
      - "curl": min elbow angle indicates curl ROM
    """
    if mode == "curl":
        if min_angle <= 70:
            depth = "Full"
            quality = "good"
            feedback = "Good range of motion — strong curl."
        elif min_angle <= 95:
            depth = "Partial"
            quality = "partial"
            feedback = "Slightly partial curl — try to squeeze to the top safely."
        else:
            depth = "Very Partial"
            quality = "incorrect"
            feedback = "Curl range of motion too small — slow down and complete the rep."
    else:
        # squat-like default
        if min_angle <= 90:
            depth = "Full"
            quality = "good"
            feedback = "Good depth and control."
        elif min_angle <= 110:
            depth = "Shallow"
            quality = "partial"
            feedback = "Slightly shallow depth — try to reach a bit deeper safely."
        else:
            depth = "Very Shallow"
            quality = "incorrect"
            feedback = "Depth too shallow — focus on range of motion and stability."

    seconds = frames_in_rep / fps if fps > 0 else 0
    if seconds >= 2.2:
        tempo = "Controlled"
    elif seconds >= 1.2:
        tempo = "Normal"
    else:
        tempo = "Fast"

    stability = "Stable" if quality != "incorrect" else "Unstable"
    return quality, feedback, tempo, depth, stability


def count_reps_from_angle_series(
    angles: List[float],
    fps: float = 10.0,
    min_drop_deg: float = 12.0,
    min_rep_frames: int = 6,
    mode: str = "squat",
) -> List[Rep]:
    """
    Very lightweight rep counting from an angle time series.

    We detect reps as: start near a local max -> descend to local min -> ascend back above a threshold.
    """
    if len(angles) < 3:
        return []

    y = _smooth(angles, window=5)

    # Establish a baseline top position (standing) from upper quartile.
    top_candidates = sorted(y)[max(0, int(len(y) * 0.7)) :]
    top_level = _median(top_candidates) if top_candidates else _median(y)

    # State machine
    reps: List[Rep] = []
    state = "top"  # top -> down -> up
    rep_start_idx: Optional[int] = None
    min_idx: Optional[int] = None
    min_val = 1e9

    # Thresholds for hysteresis
    down_trigger = top_level - max(6.0, min_drop_deg * 0.35)
    up_trigger = top_level - max(3.0, min_drop_deg * 0.15)

    for i, v in enumerate(y):
        if state == "top":
            # Wait for clear descent
            if v < down_trigger:
                state = "down"
                rep_start_idx = i
                min_idx = i
                min_val = v
        elif state == "down":
            if v < min_val:
                min_val = v
                min_idx = i
            # Start rising meaningfully: switch to up when we pass min + small epsilon
            if min_idx is not None and i - min_idx >= 1 and v > min_val + 2.0:
                state = "up"
        elif state == "up":
            # Rep completes when we come back up near the top
            if v >= up_trigger:
                if rep_start_idx is not None and min_idx is not None:
                    frames_in_rep = i - rep_start_idx + 1
                    if frames_in_rep >= min_rep_frames and (top_level - min_val) >= min_drop_deg:
                        rep_number = len(reps) + 1
                        quality, feedback, tempo, depth, stability = _classify_rep(min_val, frames_in_rep, fps, mode)
                        reps.append(
                            Rep(
                                rep_number=rep_number,
                                quality=quality,
                                feedback=feedback,
                                tempo=tempo,
                                depth=depth,
                                stability=stability,
                            )
                        )
                # Reset for next rep
                state = "top"
                rep_start_idx = None
                min_idx = None
                min_val = 1e9

    return reps


def count_reps_from_knee_angles(
    knee_angles: List[float],
    fps: float = 10.0,
    min_drop_deg: float = 12.0,
    min_rep_frames: int = 6,
) -> List[Rep]:
    # Backwards-compatible wrapper.
    return count_reps_from_angle_series(
        knee_angles,
        fps=fps,
        min_drop_deg=min_drop_deg,
        min_rep_frames=min_rep_frames,
        mode="squat",
    )

