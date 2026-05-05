"""
services/form_rules.py — Biomechanical Form Rules (22 Exercises)
=================================================================
Rule-based engine that checks whether an exercise is being performed
correctly using joint angles from MediaPipe.

Completely separate from the BiLSTM — the model classifies WHAT exercise
is happening; this module checks HOW it is being done.

Returns a FormResult with:
    form_correct  : bool
    violations    : list of violation type strings
    feedback_msgs : human-readable correction for each violation
    severity      : "ok" | "warning" | "error"
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class FormResult:
    form_correct:  bool       = True
    violations:    List[str]  = field(default_factory=list)
    feedback_msgs: List[str]  = field(default_factory=list)
    severity:      str        = "ok"    # ok | warning | error

    def add(self, violation: str, message: str, severity: str = "warning"):
        self.form_correct = False
        self.violations.append(violation)
        self.feedback_msgs.append(message)
        self.severity = "error" if ("error" in [severity, self.severity]) else "warning"


Angles = Dict[str, float]


# ══════════════════════════════════════════════════════════════════════════
# EXERCISE RULE FUNCTIONS  (one per exercise)
# ══════════════════════════════════════════════════════════════════════════

def _squat(a: Angles) -> FormResult:
    r = FormResult()
    lk, rk  = a.get("left_knee_angle", 180),  a.get("right_knee_angle", 180)
    torso    = a.get("torso_angle", 180)
    avg_knee = (lk + rk) / 2

    if abs(lk - rk) > 15:
        r.add("knee_asymmetry", "Keep both knees tracking evenly — one knee is caving in")
    if torso < 130:
        r.add("back_rounding", "Chest up — your torso is leaning too far forward", "error")
    if avg_knee > 130:
        r.add("shallow_depth", "Squat deeper — aim for thighs parallel to the floor")
    return r


def _pushup(a: Angles) -> FormResult:
    r = FormResult()
    le, re   = a.get("left_elbow_angle", 180), a.get("right_elbow_angle", 180)
    torso    = a.get("torso_angle", 180)

    if torso < 155:
        r.add("hip_sag", "Keep your body in a straight line — hips are dropping", "error")
    if abs(le - re) > 20:
        r.add("elbow_asymmetry", "Both arms should push equally — check elbow symmetry")
    return r


def _deadlift(a: Angles) -> FormResult:
    r = FormResult()
    torso = a.get("torso_angle", 180)
    lk    = a.get("left_knee_angle", 180)
    rk    = a.get("right_knee_angle", 180)

    if torso < 130:
        r.add("back_rounding", "Maintain a neutral spine — avoid rounding your lower back", "error")
    if abs(lk - rk) > 20:
        r.add("knee_asymmetry", "Keep both knees tracking symmetrically")
    return r


def _romanian_deadlift(a: Angles) -> FormResult:
    r = FormResult()
    torso    = a.get("torso_angle", 180)
    lh, rh   = a.get("left_hip_angle", 180), a.get("right_hip_angle", 180)

    if torso < 120:
        r.add("back_rounding", "Keep your back flat — hinge from the hips, not the spine", "error")
    if abs(lh - rh) > 20:
        r.add("hip_asymmetry", "Both hips should lower evenly — avoid twisting")
    return r


def _hammer_curl(a: Angles) -> FormResult:
    r = FormResult()
    ls, rs   = a.get("left_shoulder_angle", 45),  a.get("right_shoulder_angle", 45)
    le, re   = a.get("left_elbow_angle", 180),     a.get("right_elbow_angle", 180)

    if ls > 60 or rs > 60:
        r.add("elbow_swinging", "Keep elbows tight to your sides — avoid swinging forward")
    if abs(le - re) > 25:
        r.add("elbow_asymmetry", "Both arms should curl to the same height")
    return r


def _barbell_biceps_curl(a: Angles) -> FormResult:
    r = FormResult()
    ls, rs = a.get("left_shoulder_angle", 45), a.get("right_shoulder_angle", 45)
    le, re = a.get("left_elbow_angle", 180),   a.get("right_elbow_angle", 180)

    if ls > 55 or rs > 55:
        r.add("elbow_flare", "Keep elbows pinned at your sides — avoid swinging")
    if abs(le - re) > 20:
        r.add("wrist_asymmetry", "Keep the bar level — both arms should move together")
    return r


def _lateral_raise(a: Angles) -> FormResult:
    r = FormResult()
    ls, rs = a.get("left_shoulder_angle", 20), a.get("right_shoulder_angle", 20)
    le, re = a.get("left_elbow_angle", 150),   a.get("right_elbow_angle", 150)

    if abs(ls - rs) > 20:
        r.add("shoulder_asymmetry", "Raise both arms symmetrically to shoulder height")
    if le < 120 or re < 120:
        r.add("elbow_locked", "Keep a slight bend in your elbows throughout the movement")
    return r


def _shoulder_press(a: Angles) -> FormResult:
    r = FormResult()
    le, re = a.get("left_elbow_angle", 90), a.get("right_elbow_angle", 90)
    torso  = a.get("torso_angle", 180)

    if abs(le - re) > 20:
        r.add("press_asymmetry", "Both arms should press to the same height simultaneously")
    if torso < 160:
        r.add("back_arch", "Keep your core tight — avoid excessive lower back arch", "error")
    return r


def _bench_press(a: Angles) -> FormResult:
    r = FormResult()
    le, re = a.get("left_elbow_angle", 90), a.get("right_elbow_angle", 90)
    ls, rs = a.get("left_shoulder_angle", 70), a.get("right_shoulder_angle", 70)

    if abs(le - re) > 20:
        r.add("press_asymmetry", "Keep the bar level — both arms pressing evenly")
    if ls > 90 or rs > 90:
        r.add("elbow_flare", "Tuck elbows slightly — avoid flaring them out 90° from body", "warning")
    return r


def _incline_bench_press(a: Angles) -> FormResult:
    return _bench_press(a)   # Same rules, different angle of body


def _decline_bench_press(a: Angles) -> FormResult:
    return _bench_press(a)


def _chest_fly_machine(a: Angles) -> FormResult:
    r = FormResult()
    le, re = a.get("left_elbow_angle", 150), a.get("right_elbow_angle", 150)

    if abs(le - re) > 25:
        r.add("arc_asymmetry", "Both arms should follow the same arc width")
    if le < 100 or re < 100:
        r.add("elbow_bent_too_much", "Keep arms almost straight with just a slight elbow bend")
    return r


def _lat_pulldown(a: Angles) -> FormResult:
    r = FormResult()
    le, re = a.get("left_elbow_angle", 160), a.get("right_elbow_angle", 160)
    torso  = a.get("torso_angle", 180)

    if abs(le - re) > 20:
        r.add("pull_asymmetry", "Pull evenly — both elbows should come down symmetrically")
    if torso < 140:
        r.add("excessive_lean", "Avoid leaning back too far — slight lean is fine, not 45°")
    return r


def _pull_up(a: Angles) -> FormResult:
    r = FormResult()
    le, re = a.get("left_elbow_angle", 90), a.get("right_elbow_angle", 90)
    torso  = a.get("torso_angle", 170)

    if abs(le - re) > 25:
        r.add("pull_asymmetry", "Pull evenly with both arms — avoid one side dominating")
    if torso < 150:
        r.add("swinging", "Keep your core tight — avoid swinging or kipping")
    return r


def _t_bar_row(a: Angles) -> FormResult:
    r = FormResult()
    torso = a.get("torso_angle", 180)
    le, re = a.get("left_elbow_angle", 90), a.get("right_elbow_angle", 90)

    if torso < 120:
        r.add("back_rounding", "Keep your back flat during the row — don't round your spine", "error")
    if abs(le - re) > 20:
        r.add("pull_asymmetry", "Pull evenly with both arms")
    return r


def _hip_thrust(a: Angles) -> FormResult:
    r = FormResult()
    lh, rh = a.get("left_hip_angle", 160), a.get("right_hip_angle", 160)
    lk, rk = a.get("left_knee_angle", 90), a.get("right_knee_angle", 90)

    avg_hip = (lh + rh) / 2
    if avg_hip < 140:
        r.add("incomplete_extension", "Fully extend your hips at the top — squeeze your glutes")
    if abs(lk - rk) > 20:
        r.add("knee_asymmetry", "Keep both knees at the same angle — check foot position")
    return r


def _leg_extension(a: Angles) -> FormResult:
    r = FormResult()
    lk, rk = a.get("left_knee_angle", 90), a.get("right_knee_angle", 90)

    if abs(lk - rk) > 20:
        r.add("knee_asymmetry", "Extend both legs to the same height simultaneously")
    avg_knee = (lk + rk) / 2
    if avg_knee < 150:
        r.add("incomplete_extension", "Fully extend your knees at the top of the movement")
    return r


def _leg_raises(a: Angles) -> FormResult:
    r = FormResult()
    lh, rh  = a.get("left_hip_angle", 90), a.get("right_hip_angle", 90)
    torso   = a.get("torso_angle", 170)

    if abs(lh - rh) > 20:
        r.add("hip_asymmetry", "Raise both legs evenly — avoid one hip dropping lower")
    if torso < 155:
        r.add("back_lift", "Keep your lower back pressed to the bench throughout", "warning")
    return r


def _plank(a: Angles) -> FormResult:
    r = FormResult()
    torso    = a.get("torso_angle", 180)
    lh       = a.get("left_hip_angle", 180)

    if torso < 155:
        r.add("hip_sag", "Engage your core — hips are dropping below the line", "error")
    if lh < 155:
        r.add("hip_pike", "Lower your hips — body should form a straight line", "warning")
    return r


def _russian_twist(a: Angles) -> FormResult:
    r = FormResult()
    torso = a.get("torso_angle", 135)
    lk, rk = a.get("left_knee_angle", 90), a.get("right_knee_angle", 90)

    if not (110 < torso < 165):
        r.add("torso_angle", "Lean back at ~45° with chest lifted — not flat on the ground")
    if lk > 120 or rk > 120:
        r.add("knees_not_bent", "Bend your knees to ~90° for better core engagement")
    return r


def _tricep_dips(a: Angles) -> FormResult:
    r = FormResult()
    le, re = a.get("left_elbow_angle", 90), a.get("right_elbow_angle", 90)
    torso  = a.get("torso_angle", 170)

    if abs(le - re) > 20:
        r.add("elbow_asymmetry", "Both elbows should bend to the same angle evenly")
    if torso < 150:
        r.add("torso_lean", "Keep your torso more upright — avoid excessive forward lean")
    return r


def _tricep_pushdown(a: Angles) -> FormResult:
    r = FormResult()
    le, re = a.get("left_elbow_angle", 90), a.get("right_elbow_angle", 90)
    ls, rs = a.get("left_shoulder_angle", 10), a.get("right_shoulder_angle", 10)

    if ls > 40 or rs > 40:
        r.add("elbow_flare", "Keep elbows pinned to your sides — avoid letting them drift forward")
    if abs(le - re) > 20:
        r.add("push_asymmetry", "Both arms should push down to full extension together")
    return r


# ══════════════════════════════════════════════════════════════════════════
# REGISTRY  —  exercise name → rule function
# ══════════════════════════════════════════════════════════════════════════

EXERCISE_RULES: Dict[str, callable] = {
    "squat":               _squat,
    "pushup":              _pushup,
    "deadlift":            _deadlift,
    "romanian_deadlift":   _romanian_deadlift,
    "hammer_curl":         _hammer_curl,
    "barbell_biceps_curl": _barbell_biceps_curl,
    "lateral_raise":       _lateral_raise,
    "shoulder_press":      _shoulder_press,
    "bench_press":         _bench_press,
    "incline_bench_press": _incline_bench_press,
    "decline_bench_press": _decline_bench_press,
    "chest_fly_machine":   _chest_fly_machine,
    "lat_pulldown":        _lat_pulldown,
    "pull_up":             _pull_up,
    "t_bar_row":           _t_bar_row,
    "hip_thrust":          _hip_thrust,
    "leg_extension":       _leg_extension,
    "leg_raises":          _leg_raises,
    "plank":               _plank,
    "russian_twist":       _russian_twist,
    "tricep_dips":         _tricep_dips,
    "tricep_pushdown":     _tricep_pushdown,
}

# These classes skip form checking (detected but not exercises)
NON_EXERCISE = {
    "No Detection", "Warming up...", "No Exercise Detected", "Model Not Loaded"
}


def check_form(exercise: str, angles: Angles) -> FormResult:
    """
    Main entry point called from the router for every frame.
    Returns a FormResult for the given exercise + angles.
    """
    if not angles or exercise in NON_EXERCISE:
        return FormResult()

    rule_fn = EXERCISE_RULES.get(exercise)
    if rule_fn is None:
        return FormResult()   # unknown exercise → pass through

    return rule_fn(angles)