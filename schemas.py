from pydantic import BaseModel
from typing import List, Optional
from datetime import date, time


# ---------------------------
# USER / SESSION
# ---------------------------

class SessionCreate(BaseModel):
    name: str
    email: str
    session_date: Optional[str] = None
    capture_time: Optional[str] = None


# ---------------------------
# FRAME DATA
# ---------------------------

class Frame(BaseModel):

    elbow_angle: float
    hip_angle: float
    knee_angle: float

    shoulder_width: float
    hip_width: float
    torso_leg_ratio: float

    body_angle: float
    posture_label: Optional[str] = None


class FramesUpload(BaseModel):
    session_id: int
    frames: List[Frame]


# ---------------------------
# ANALYSIS
# ---------------------------

class AnalysisRequest(BaseModel):
    session_id: int


# ---------------------------
# WORKOUT
# ---------------------------

class WorkoutRequest(BaseModel):
    session_id: int