"""
services/rep_counter.py — Rep Counting State Machine (22 Exercises)
====================================================================
Each exercise has two states:
    "start" → resting / top position
    "mid"   → bottom / contracted position

A rep is counted when: start → mid → start  (+1 rep)

Plank is a hold — no reps, state = "hold" while form maintained.

RepCounter is instantiated ONCE per active exercise session
and discarded when the session ends.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ExerciseState:
    """Tracks state + rep count for one exercise within a session."""
    state: str = "start"
    reps:  int = 0


class RepCounter:
    """
    Manages rep counting for all exercises in one ExerciseSession.
    Create one instance per active session.
    """

    def __init__(self):
        self._states: Dict[str, ExerciseState] = {}

    def update(self, exercise: str, angles: Dict[str, float]) -> Tuple[int, str]:
        """
        Process one frame for the given exercise.
        Returns: (current_rep_count, current_state)
        """
        if exercise not in self._states:
            self._states[exercise] = ExerciseState()

        obj = self._states[exercise]
        fn  = COUNTER_REGISTRY.get(exercise)
        if fn is None:
            return obj.reps, obj.state

        new_state, rep_done = fn(angles, obj.state)
        obj.state = new_state
        if rep_done:
            obj.reps += 1

        return obj.reps, obj.state

    def get_reps(self, exercise: str) -> int:
        return self._states.get(exercise, ExerciseState()).reps

    def get_all(self) -> Dict[str, int]:
        return {ex: s.reps for ex, s in self._states.items()}

    def total(self) -> int:
        return sum(s.reps for s in self._states.values())


# ── Individual state machines ─────────────────────────────────────────────
# Returns (new_state: str, rep_completed: bool)

def _squat_c(a, state):
    avg = (a.get("left_knee_angle", 180) + a.get("right_knee_angle", 180)) / 2
    if state == "start" and avg <= 110: return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _pushup_c(a, state):
    avg = (a.get("left_elbow_angle", 180) + a.get("right_elbow_angle", 180)) / 2
    if state == "start" and avg <= 100: return "mid", False
    if state == "mid"   and avg >= 150: return "start", True
    return state, False

def _deadlift_c(a, state):
    hip = a.get("left_hip_angle", 180)
    if state == "start" and hip <= 100: return "mid", False
    if state == "mid"   and hip >= 155: return "start", True
    return state, False

def _romanian_deadlift_c(a, state):
    hip = a.get("left_hip_angle", 180)
    if state == "start" and hip <= 110: return "mid", False
    if state == "mid"   and hip >= 160: return "start", True
    return state, False

def _curl_c(a, state):
    avg = (a.get("left_elbow_angle", 180) + a.get("right_elbow_angle", 180)) / 2
    if state == "start" and avg <= 70:  return "mid", False
    if state == "mid"   and avg >= 145: return "start", True
    return state, False

def _lateral_raise_c(a, state):
    avg = (a.get("left_shoulder_angle", 20) + a.get("right_shoulder_angle", 20)) / 2
    if state == "start" and avg >= 60: return "mid", False
    if state == "mid"   and avg <= 25: return "start", True
    return state, False

def _overhead_press_c(a, state):
    avg = (a.get("left_elbow_angle", 90) + a.get("right_elbow_angle", 90)) / 2
    if state == "start" and avg <= 100: return "mid", False
    if state == "mid"   and avg >= 160: return "start", True
    return state, False

def _bench_press_c(a, state):
    avg = (a.get("left_elbow_angle", 180) + a.get("right_elbow_angle", 180)) / 2
    if state == "start" and avg <= 95:  return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _fly_c(a, state):
    avg = (a.get("left_elbow_angle", 90) + a.get("right_elbow_angle", 90)) / 2
    if state == "start" and avg <= 110: return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _lat_pulldown_c(a, state):
    avg = (a.get("left_elbow_angle", 180) + a.get("right_elbow_angle", 180)) / 2
    if state == "start" and avg <= 90:  return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _pull_up_c(a, state):
    avg = (a.get("left_elbow_angle", 180) + a.get("right_elbow_angle", 180)) / 2
    if state == "start" and avg <= 90:  return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _row_c(a, state):
    avg = (a.get("left_elbow_angle", 180) + a.get("right_elbow_angle", 180)) / 2
    if state == "start" and avg <= 90:  return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _hip_thrust_c(a, state):
    avg = (a.get("left_hip_angle", 90) + a.get("right_hip_angle", 90)) / 2
    if state == "start" and avg >= 150: return "mid", False
    if state == "mid"   and avg <= 110: return "start", True
    return state, False

def _leg_extension_c(a, state):
    avg = (a.get("left_knee_angle", 90) + a.get("right_knee_angle", 90)) / 2
    if state == "start" and avg >= 155: return "mid", False
    if state == "mid"   and avg <= 100: return "start", True
    return state, False

def _leg_raises_c(a, state):
    avg = (a.get("left_hip_angle", 180) + a.get("right_hip_angle", 180)) / 2
    if state == "start" and avg <= 90:  return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _plank_c(a, state):
    # Plank is a hold — no reps counted
    return "hold", False

def _russian_twist_c(a, state):
    le = a.get("left_elbow_angle", 160)
    re = a.get("right_elbow_angle", 160)
    asymmetry = abs(le - re) > 20
    if state == "start" and asymmetry:      return "mid", False
    if state == "mid"   and not asymmetry:  return "start", True
    return state, False

def _tricep_dips_c(a, state):
    avg = (a.get("left_elbow_angle", 90) + a.get("right_elbow_angle", 90)) / 2
    if state == "start" and avg <= 90:  return "mid", False
    if state == "mid"   and avg >= 155: return "start", True
    return state, False

def _tricep_pushdown_c(a, state):
    avg = (a.get("left_elbow_angle", 90) + a.get("right_elbow_angle", 90)) / 2
    if state == "start" and avg >= 155: return "mid", False
    if state == "mid"   and avg <= 90:  return "start", True
    return state, False


# ── Registry ──────────────────────────────────────────────────────────────

COUNTER_REGISTRY: Dict[str, callable] = {
    "squat":               _squat_c,
    "pushup":              _pushup_c,
    "deadlift":            _deadlift_c,
    "romanian_deadlift":   _romanian_deadlift_c,
    "hammer_curl":         _curl_c,
    "barbell_biceps_curl": _curl_c,
    "lateral_raise":       _lateral_raise_c,
    "shoulder_press":      _overhead_press_c,
    "bench_press":         _bench_press_c,
    "incline_bench_press": _bench_press_c,
    "decline_bench_press": _bench_press_c,
    "chest_fly_machine":   _fly_c,
    "lat_pulldown":        _lat_pulldown_c,
    "pull_up":             _pull_up_c,
    "t_bar_row":           _row_c,
    "hip_thrust":          _hip_thrust_c,
    "leg_extension":       _leg_extension_c,
    "leg_raises":          _leg_raises_c,
    "plank":               _plank_c,
    "russian_twist":       _russian_twist_c,
    "tricep_dips":         _tricep_dips_c,
    "tricep_pushdown":     _tricep_pushdown_c,
}