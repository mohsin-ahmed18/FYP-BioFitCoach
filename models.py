"""
models.py
==========
SQLAlchemy ORM models for ALL tables — Module 1 (existing) + Module 2 (new).

Module 1 tables  (already exist in your DB — do NOT drop/recreate):
    Session, Frame, BiomechanicalProfile, WorkoutPlan

Module 2 tables  (NEW — created by running migrate_module2.py):
    ExerciseSession, RepLog, FormFeedbackLog, SessionFeedback

Import usage:
    from models import ExerciseSession, RepLog, FormFeedbackLog, SessionFeedback
"""

from sqlalchemy import (
    Column, Integer, String, Float,
    Boolean, Text, DateTime,
)
from datetime import datetime
from database import Base


# ─────────────────────────────────────────────────────────────────
# MODULE 1 TABLES  (existing — do not modify)
# ─────────────────────────────────────────────────────────────────

class Session(Base):
    __tablename__ = "sessions"

    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String)
    email        = Column(String)
    session_date = Column(String)
    capture_time = Column(String)


class Frame(Base):
    __tablename__ = "frames"

    id              = Column(Integer, primary_key=True, index=True)
    session_id      = Column(Integer, index=True)
    elbow_angle     = Column(Float)
    hip_angle       = Column(Float)
    knee_angle      = Column(Float)
    shoulder_width  = Column(Float)
    hip_width       = Column(Float)
    torso_leg_ratio = Column(Float)
    body_angle      = Column(Float)
    posture_label   = Column(String, nullable=True)


class BiomechanicalProfile(Base):
    __tablename__ = "biomechanical_profiles"

    id                            = Column(Integer, primary_key=True, index=True)
    session_id                    = Column(Integer, index=True)
    alignment_score               = Column(Float)
    stability_score               = Column(Float)
    symmetry_score                = Column(Float)
    biomechanical_readiness_index = Column(Float)
    structural_bias               = Column(String)
    primary_limit_factor          = Column(String)


class WorkoutPlan(Base):
    __tablename__ = "workout_plans"

    id            = Column(Integer, primary_key=True, index=True)
    session_id    = Column(Integer, index=True)
    exercise_name = Column(String)
    llm_output    = Column(Text, nullable=True)


# ─────────────────────────────────────────────────────────────────
# MODULE 2 TABLES  (new)
# ─────────────────────────────────────────────────────────────────

class ExerciseSession(Base):
    """
    A live workout session (Module 2).
    Links to a Module 1 session so we can access the user's
    biomechanical profile for personalised LLM feedback.

    status: 'active' while recording | 'completed' when ended
    """
    __tablename__ = "exercise_sessions"

    id                     = Column(Integer, primary_key=True, index=True)
    session_id             = Column(Integer, index=True)      # FK → sessions.id
    started_at             = Column(DateTime, default=datetime.utcnow)
    ended_at               = Column(DateTime, nullable=True)
    status                 = Column(String, default="active")
    total_reps             = Column(Integer, default=0)
    exercises_performed    = Column(Text, nullable=True)      # JSON list e.g. '["squat","pushup"]'
    llm_feedback_generated = Column(Boolean, default=False)


class RepLog(Base):
    """
    One row = one completed repetition.

    rep_quality:
        'good'       → form_correct=True  AND confidence ≥ 0.80
        'acceptable' → form_correct=True  AND confidence < 0.80
        'poor'       → form_correct=False
    """
    __tablename__ = "rep_logs"

    id                  = Column(Integer, primary_key=True, index=True)
    exercise_session_id = Column(Integer, index=True)    # FK → exercise_sessions.id
    exercise_name       = Column(String)
    rep_number          = Column(Integer)
    rep_quality         = Column(String)                 # good | acceptable | poor
    form_correct        = Column(Boolean, default=True)
    confidence_score    = Column(Float)
    knee_angle_avg      = Column(Float, nullable=True)
    elbow_angle_avg     = Column(Float, nullable=True)
    hip_angle_avg       = Column(Float, nullable=True)
    torso_angle_avg     = Column(Float, nullable=True)
    timestamp           = Column(DateTime, default=datetime.utcnow)


class FormFeedbackLog(Base):
    """
    One row = one form violation detected during a rep.
    Multiple violations can occur in a single rep.

    severity: 'warning' (suboptimal) | 'error' (injury risk)
    """
    __tablename__ = "form_feedback_logs"

    id                  = Column(Integer, primary_key=True, index=True)
    exercise_session_id = Column(Integer, index=True)
    rep_log_id          = Column(Integer, nullable=True, index=True)
    exercise_name       = Column(String)
    violation_type      = Column(String)     # e.g. 'knee_cave', 'back_rounding'
    feedback_message    = Column(String)     # e.g. "Keep knees over toes"
    severity            = Column(String, default="warning")
    timestamp           = Column(DateTime, default=datetime.utcnow)


class SessionFeedback(Base):
    """
    LLM-generated coaching report for a completed ExerciseSession.
    Created once when the user ends their session.

    llm_summary stores full JSON:
    {
        "overall_summary": "...",
        "overall_score": 85,
        "exercise_breakdown": { "squat": {...} },
        "key_issues": [...],
        "improvements": [...],
        "next_session_tips": "..."
    }
    """
    __tablename__ = "session_feedback"

    id                  = Column(Integer, primary_key=True, index=True)
    exercise_session_id = Column(Integer, index=True, unique=True)
    llm_summary         = Column(Text)
    overall_score       = Column(Float, nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)