"""
schemas.py
===========
Pydantic request/response models for the entire API.

Module 1 schemas (existing — unchanged):
    SessionCreate, Frame, FramesUpload, AnalysisRequest, WorkoutRequest

Module 2 schemas (new):
    StartSessionRequest/Response, ProcessFrameRequest/Response,
    LogRepRequest, EndSessionRequest, and all feedback response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date, time, datetime


# ─────────────────────────────────────────────────────────────────
# MODULE 1 SCHEMAS  (existing — do not change)
# ─────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    name:         str
    email:        str
    session_date: Optional[str] = None
    capture_time: Optional[str] = None


class Frame(BaseModel):
    elbow_angle:     float
    hip_angle:       float
    knee_angle:      float
    shoulder_width:  float
    hip_width:       float
    torso_leg_ratio: float
    body_angle:      float
    posture_label:   Optional[str] = None


class FramesUpload(BaseModel):
    session_id: int
    frames:     List[Frame]


class AnalysisRequest(BaseModel):
    session_id: int


class WorkoutRequest(BaseModel):
    session_id: int


# ─────────────────────────────────────────────────────────────────
# MODULE 2 SCHEMAS  (new)
# ─────────────────────────────────────────────────────────────────

# ── Requests ────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    """
    Start a new Module 2 exercise session.
    session_id must be an existing Module 1 session — this links the
    user's biomechanical profile to their workout for personalised feedback.
    """
    session_id: int = Field(
        ...,
        description="Existing Module 1 session ID (from POST /session)",
        example=1,
    )


class ProcessFrameRequest(BaseModel):
    """
    Send one webcam frame for real-time exercise detection.
    Call this repeatedly (~10 fps) from your client.
    """
    exercise_session_id: int = Field(
        ...,
        description="Active exercise session ID from POST /exercise/start-session",
    )
    frame_b64: str = Field(
        ...,
        description="Base64-encoded JPEG or PNG frame from webcam",
    )


class LogRepRequest(BaseModel):
    """
    Save one completed repetition to the database.
    Call this when your rep counter state changes: mid → start.
    """
    exercise_session_id: int
    exercise_name:       str
    form_correct:        bool
    confidence_score:    float
    violations:          List[str]  = Field(default_factory=list)
    feedback_messages:   List[str]  = Field(default_factory=list)
    knee_angle_avg:      Optional[float] = None
    elbow_angle_avg:     Optional[float] = None
    hip_angle_avg:       Optional[float] = None
    torso_angle_avg:     Optional[float] = None


class EndSessionRequest(BaseModel):
    """End an active session and trigger LLM feedback generation."""
    exercise_session_id: int


# ── Responses ───────────────────────────────────────────────────

class StartSessionResponse(BaseModel):
    exercise_session_id: int
    message:             str


class ProcessFrameResponse(BaseModel):
    """
    Returned after every frame submission.
    buffer_ready: False while the model is warming up (first 30 frames).
                  Always show the user a "warming up..." indicator until True.
    """
    person_detected:  bool
    exercise:         str
    confidence:       float
    reps:             int
    form_correct:     bool
    feedback_msgs:    List[str]
    buffer_ready:     bool


class LogRepResponse(BaseModel):
    message:     str
    rep_id:      int
    rep_quality: str


class RepSummaryItem(BaseModel):
    rep_number:   int
    exercise:     str
    quality:      str
    form_correct: bool
    confidence:   float
    timestamp:    Optional[datetime]


class ExerciseSessionSummary(BaseModel):
    exercise_session_id: int
    status:              str
    started_at:          Optional[datetime]
    ended_at:            Optional[datetime]
    total_reps:          int
    exercises:           List[str]
    feedback:            Optional[Dict]
    overall_score:       Optional[float]


class ViolationItem(BaseModel):
    violation:  str
    message:    str
    severity:   str
    timestamp:  Optional[datetime]


class FormViolationsResponse(BaseModel):
    exercise_session_id: int
    total_violations:    int
    by_exercise:         Dict[str, List[ViolationItem]]


class WorkoutHistoryItem(BaseModel):
    exercise_session_id: int
    status:              str
    started_at:          Optional[datetime]
    ended_at:            Optional[datetime]
    total_reps:          int
    exercises:           List[str]
    overall_score:       Optional[float]


class WorkoutHistoryResponse(BaseModel):
    session_id: int
    total:      int
    workouts:   List[WorkoutHistoryItem]


class EndSessionResponse(BaseModel):
    message:    str
    total_reps: int
    exercises:  List[str]
    feedback:   Dict