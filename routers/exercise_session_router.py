"""
routers/exercise_session_router.py — Module 2 Live Workout Endpoints
=====================================================================
Handles the full lifecycle of a live workout session:

    POST /exercise/start-session        Create session, init frame buffer
    POST /exercise/process-frame        Send one frame → get detection + reps + form
    POST /exercise/log-rep              Save a confirmed rep to the database
    POST /exercise/end-session          End session → trigger LLM feedback
    GET  /exercise/session/{id}         Get session summary + feedback
    GET  /exercise/session/{id}/reps    Get all rep logs for a session

Import paths use flat project structure (everything at project root level).
"""

import base64
import json
import numpy as np
import cv2
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db
from models import (
    ExerciseSession,
    RepLog,
    FormFeedbackLog,
    SessionFeedback,
)
from schemas import (
    StartExerciseSessionRequest,
    StartExerciseSessionResponse,
    ProcessFrameRequest,
    ProcessFrameResponse,
    LogRepRequest,
    LogRepResponse,
    EndSessionRequest,
    ExerciseSessionSummary,
    SessionRepsResponse,
    RepDetail,
)
from services.exercise_detector import detector
from services.form_rules import check_form, NON_EXERCISE
from services.rep_counter import RepCounter
from services.rep_llm_feedback import generate_session_feedback

router = APIRouter(prefix="/exercise", tags=["Module 2 — Exercise Coach"])

# ── In-memory rep counters ────────────────────────────────────────────────
# key = str(exercise_session_id)  →  RepCounter instance
# Lives only while the session is active. Cleared on end-session.
_counters: dict = {}


# ── Helper ────────────────────────────────────────────────────────────────

def _decode_frame(b64: str) -> np.ndarray:
    """
    Decode a base64-encoded JPEG or PNG string into a BGR numpy array.

    Strips the 'data:image/jpeg;base64,' prefix automatically if present
    (browsers add this when using the Canvas API).
    """
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw   = base64.b64decode(b64)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None — invalid image bytes")
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {e}")


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Start Session
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/start-session",
    response_model=StartExerciseSessionResponse,
    status_code=201,
    summary="Create a new live workout session",
    description=(
        "Links to an existing Module 1 session (body-scan session) so the system "
        "can personalise post-workout feedback using the user's biomechanical profile. "
        "Returns an exercise_session_id — pass this in every subsequent call."
    ),
)
def start_session(
    body: StartExerciseSessionRequest,
    db:   Session = Depends(get_db),
):
    # Verify the Module 1 session exists
    row = db.execute(
        text("SELECT id FROM sessions WHERE id = :sid"),
        {"sid": body.session_id},
    ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Module 1 session {body.session_id} not found. "
                   "Create one via POST /session first.",
        )

    # Create the exercise session record
    ex_sess = ExerciseSession(
        session_id=body.session_id,
        started_at=datetime.utcnow(),
        status="active",
        total_reps=0,
    )
    db.add(ex_sess)
    db.commit()
    db.refresh(ex_sess)

    # Initialise the BiLSTM frame buffer and rep counter for this session
    key = str(ex_sess.id)
    detector.init_session(key)
    _counters[key] = RepCounter()

    return StartExerciseSessionResponse(
        exercise_session_id=ex_sess.id,
        message=(
            f"Workout session {ex_sess.id} started successfully. "
            "Send webcam frames to POST /exercise/process-frame"
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Process Frame
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/process-frame",
    response_model=ProcessFrameResponse,
    summary="Send one webcam frame and get live detection result",
    description=(
        "Call this endpoint in a loop from your client (e.g. 10–15 times per second). "
        "Returns the detected exercise, rep count, and form feedback for each frame. "
        "The first 30 frames warm up the BiLSTM buffer — "
        "exercise shows 'Warming up...' and buffer_ready=False during this time."
    ),
)
def process_frame(
    body: ProcessFrameRequest,
    db:   Session = Depends(get_db),
):
    key = str(body.exercise_session_id)

    # Verify session is active (lightweight check — just query status)
    ex_sess = db.query(ExerciseSession).filter(
        ExerciseSession.id     == body.exercise_session_id,
        ExerciseSession.status == "active",
    ).first()

    if not ex_sess:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Active exercise session {body.exercise_session_id} not found. "
                "Start one via POST /exercise/start-session."
            ),
        )

    # Decode base64 → BGR numpy array
    frame = _decode_frame(body.frame_b64)

    # Run BiLSTM inference through the detector
    result   = detector.process_frame(frame, key)
    exercise = result["exercise"]

    # Rep counting (only for real exercises while buffer is ready)
    reps    = 0
    counter = _counters.get(key)
    if (
        result["person_detected"]
        and result["buffer_ready"]
        and exercise not in NON_EXERCISE
        and counter is not None
    ):
        reps, _ = counter.update(exercise, result["rule_angles"])

    # Biomechanical form checking
    form_result = check_form(exercise, result["rule_angles"])

    return ProcessFrameResponse(
        person_detected=result["person_detected"],
        exercise=exercise,
        confidence=result["confidence"],
        reps=reps,
        form_correct=form_result.form_correct,
        feedback_msgs=form_result.feedback_msgs,
        buffer_ready=result["buffer_ready"],
        buffer_status=result["buffer_status"],
    )


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Log Rep
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/log-rep",
    response_model=LogRepResponse,
    summary="Save a completed rep to the database",
    description=(
        "Call this each time your rep counter increments (state goes mid → start). "
        "Rep quality is auto-determined: "
        "good (form_correct=True + confidence≥0.80), "
        "acceptable (form_correct=True + confidence<0.80), "
        "poor (form_correct=False)."
    ),
)
def log_rep(
    body: LogRepRequest,
    db:   Session = Depends(get_db),
):
    # Verify session exists and is still active
    ex_sess = db.query(ExerciseSession).filter(
        ExerciseSession.id     == body.exercise_session_id,
        ExerciseSession.status == "active",
    ).first()

    if not ex_sess:
        raise HTTPException(
            status_code=404,
            detail=f"Active exercise session {body.exercise_session_id} not found.",
        )

    # Sequential rep number for this exercise within this session
    existing_count = db.query(RepLog).filter(
        RepLog.exercise_session_id == body.exercise_session_id,
        RepLog.exercise_name       == body.exercise_name,
    ).count()

    # Auto-determine quality
    if not body.form_correct:
        quality = "poor"
    elif body.confidence_score >= 0.80:
        quality = "good"
    else:
        quality = "acceptable"

    # Save the rep
    rep = RepLog(
        exercise_session_id=body.exercise_session_id,
        exercise_name=body.exercise_name,
        rep_number=existing_count + 1,
        rep_quality=quality,
        form_correct=body.form_correct,
        confidence_score=body.confidence_score,
        knee_angle_avg=body.knee_angle_avg,
        elbow_angle_avg=body.elbow_angle_avg,
        hip_angle_avg=body.hip_angle_avg,
        torso_angle_avg=body.torso_angle_avg,
        timestamp=datetime.utcnow(),
    )
    db.add(rep)
    db.flush()   # flush to get rep.id before adding violations

    # Save any form violations for this rep
    for i, violation in enumerate(body.violations):
        msg = body.feedback_messages[i] if i < len(body.feedback_messages) else ""
        db.add(FormFeedbackLog(
            exercise_session_id=body.exercise_session_id,
            rep_log_id=rep.id,
            exercise_name=body.exercise_name,
            violation_type=violation,
            feedback_message=msg,
            severity="error" if not body.form_correct else "warning",
            timestamp=datetime.utcnow(),
        ))

    db.commit()

    return LogRepResponse(
        message="Rep logged successfully",
        rep_id=rep.id,
        rep_number=rep.rep_number,
        rep_quality=quality,
    )


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — End Session
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/end-session",
    summary="End a workout session and generate AI coaching feedback",
    description=(
        "Marks the session as completed, aggregates all rep logs and form violations, "
        "fetches the user's Module 1 biomechanical profile, and calls OpenAI to generate "
        "a personalised coaching report. Saves the report and returns it immediately."
    ),
)
def end_session(
    body: EndSessionRequest,
    db:   Session = Depends(get_db),
):
    # Verify active session
    ex_sess = db.query(ExerciseSession).filter(
        ExerciseSession.id     == body.exercise_session_id,
        ExerciseSession.status == "active",
    ).first()

    if not ex_sess:
        raise HTTPException(
            status_code=404,
            detail=f"Active exercise session {body.exercise_session_id} not found.",
        )

    # ── Step 1: Mark as completed ─────────────────────────────────────────
    ex_sess.ended_at = datetime.utcnow()
    ex_sess.status   = "completed"

    # ── Step 2: Aggregate reps and violations ─────────────────────────────
    all_reps  = db.query(RepLog).filter(
        RepLog.exercise_session_id == body.exercise_session_id
    ).all()

    all_viols = db.query(FormFeedbackLog).filter(
        FormFeedbackLog.exercise_session_id == body.exercise_session_id
    ).all()

    exercises_done            = list({r.exercise_name for r in all_reps})
    ex_sess.total_reps        = len(all_reps)
    ex_sess.exercises_performed = json.dumps(exercises_done)
    db.flush()

    # Build rep summary per exercise
    rep_summary = []
    for ex in exercises_done:
        ex_reps  = [r for r in all_reps if r.exercise_name == ex]
        good     = sum(1 for r in ex_reps if r.rep_quality == "good")
        poor     = sum(1 for r in ex_reps if r.rep_quality == "poor")
        avg_conf = (
            sum(r.confidence_score for r in ex_reps) / len(ex_reps)
            if ex_reps else 0.0
        )
        rep_summary.append({
            "exercise":       ex,
            "total_reps":     len(ex_reps),
            "good_reps":      good,
            "poor_reps":      poor,
            "avg_confidence": avg_conf,
        })

    # De-duplicate violations (count occurrences)
    seen: dict = {}
    for v in all_viols:
        k = (v.exercise_name, v.violation_type)
        if k not in seen:
            seen[k] = {
                "exercise":         v.exercise_name,
                "violation_type":   v.violation_type,
                "feedback_message": v.feedback_message,
                "count":            0,
            }
        seen[k]["count"] += 1
    viol_summary = list(seen.values())

    # ── Step 3: Fetch Module 1 biomechanical profile ──────────────────────
    bio_row = db.execute(
        text("""
            SELECT alignment_score, stability_score, symmetry_score,
                   structural_bias, primary_limit_factor
            FROM   biomechanical_profiles
            WHERE  session_id = :sid
            ORDER  BY id DESC
            LIMIT  1
        """),
        {"sid": ex_sess.session_id},
    ).fetchone()

    bio_profile = dict(bio_row._mapping) if bio_row else None

    # Fetch Module 1 workout plan (for personalised feedback context)
    plan_row = db.execute(
        text("""
            SELECT llm_output
            FROM   workout_plans
            WHERE  session_id = :sid
            ORDER  BY id DESC
            LIMIT  1
        """),
        {"sid": ex_sess.session_id},
    ).fetchone()

    workout_plan = plan_row[0] if plan_row else None

    # ── Step 4: Generate LLM coaching feedback ────────────────────────────
    feedback = generate_session_feedback(
        rep_summary=rep_summary,
        form_violations=viol_summary,
        bio_profile=bio_profile,
        workout_plan=workout_plan,
    )

    # ── Step 5: Save feedback ─────────────────────────────────────────────
    db.add(SessionFeedback(
        exercise_session_id=body.exercise_session_id,
        llm_summary=json.dumps(feedback),
        overall_score=float(feedback.get("overall_score", 70)),
        created_at=datetime.utcnow(),
    ))
    ex_sess.llm_feedback_generated = True
    db.commit()

    # ── Step 6: Clean up in-memory state ─────────────────────────────────
    key = str(body.exercise_session_id)
    detector.clear_session(key)
    _counters.pop(key, None)

    return {
        "message":    "Session completed and feedback generated successfully",
        "session_id": ex_sess.session_id,
        "total_reps": ex_sess.total_reps,
        "exercises":  exercises_done,
        "feedback":   feedback,
    }


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — Get Session Summary
# ══════════════════════════════════════════════════════════════════════════

@router.get(
    "/session/{exercise_session_id}",
    response_model=ExerciseSessionSummary,
    summary="Get session summary and AI feedback",
)
def get_session(
    exercise_session_id: int,
    db: Session = Depends(get_db),
):
    """Get the full summary and coaching feedback for any session (active or completed)."""
    es = db.query(ExerciseSession).filter(
        ExerciseSession.id == exercise_session_id
    ).first()

    if not es:
        raise HTTPException(
            status_code=404,
            detail=f"Exercise session {exercise_session_id} not found.",
        )

    fb = db.query(SessionFeedback).filter(
        SessionFeedback.exercise_session_id == exercise_session_id
    ).first()

    return ExerciseSessionSummary(
        exercise_session_id=es.id,
        status=es.status,
        started_at=str(es.started_at) if es.started_at else None,
        ended_at=str(es.ended_at)     if es.ended_at   else None,
        total_reps=es.total_reps or 0,
        exercises=json.loads(es.exercises_performed) if es.exercises_performed else [],
        feedback=json.loads(fb.llm_summary)          if fb                     else None,
        overall_score=fb.overall_score               if fb                     else None,
    )


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 6 — Get Session Reps
# ══════════════════════════════════════════════════════════════════════════

@router.get(
    "/session/{exercise_session_id}/reps",
    response_model=SessionRepsResponse,
    summary="Get all rep logs for a session",
)
def get_session_reps(
    exercise_session_id: int,
    db: Session = Depends(get_db),
):
    """Returns every rep logged during the session with quality and form details."""
    reps = db.query(RepLog).filter(
        RepLog.exercise_session_id == exercise_session_id
    ).order_by(RepLog.timestamp).all()

    return SessionRepsResponse(
        exercise_session_id=exercise_session_id,
        total_reps=len(reps),
        reps=[
            RepDetail(
                rep_number=r.rep_number,
                exercise=r.exercise_name,
                quality=r.rep_quality,
                form_correct=r.form_correct,
                confidence=r.confidence_score,
                timestamp=str(r.timestamp) if r.timestamp else None,
            )
            for r in reps
        ],
    )