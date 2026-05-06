"""
routers/rep_feedback_router.py — Feedback & History Endpoints
=============================================================
Read-only endpoints for querying coaching feedback and workout history.

    GET /feedback/session/{id}              Full LLM coaching report
    GET /feedback/session/{id}/violations   All form violations
    GET /feedback/history/{session_id}      All workout sessions for a user
    GET /feedback/summary/{session_id}      Aggregate stats across all sessions
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import ExerciseSession, SessionFeedback, FormFeedbackLog, RepLog
from schemas import (
    FormViolationsResponse,
    ViolationDetail,
    WorkoutHistoryResponse,
    WorkoutHistoryItem,
)

router = APIRouter(prefix="/feedback", tags=["Module 2 — Feedback & History"])


@router.get("/session/{exercise_session_id}")
def get_session_feedback(
    exercise_session_id: int,
    db: Session = Depends(get_db),
):
    """
    Get the full LLM coaching report for a completed exercise session.
    Only available after POST /exercise/end-session has been called.
    """
    fb = db.query(SessionFeedback).filter(
        SessionFeedback.exercise_session_id == exercise_session_id
    ).first()

    if not fb:
        raise HTTPException(
            404,
            f"No feedback found for exercise session {exercise_session_id}. "
            "Call POST /exercise/end-session first to generate it.",
        )

    return {
        "exercise_session_id": exercise_session_id,
        "overall_score":       fb.overall_score,
        "generated_at":        str(fb.created_at),
        "feedback":            json.loads(fb.llm_summary),
    }


@router.get("/session/{exercise_session_id}/violations",
            response_model=FormViolationsResponse)
def get_violations(
    exercise_session_id: int,
    db: Session = Depends(get_db),
):
    """
    Get all form violations logged during a session, grouped by exercise.
    Useful for showing the user exactly where their form broke down.
    """
    logs = db.query(FormFeedbackLog).filter(
        FormFeedbackLog.exercise_session_id == exercise_session_id
    ).order_by(FormFeedbackLog.timestamp).all()

    grouped: dict = {}
    for v in logs:
        if v.exercise_name not in grouped:
            grouped[v.exercise_name] = []
        grouped[v.exercise_name].append(
            ViolationDetail(
                violation=v.violation_type,
                message=v.feedback_message,
                severity=v.severity,
                timestamp=str(v.timestamp) if v.timestamp else None,
            )
        )

    return FormViolationsResponse(
        exercise_session_id=exercise_session_id,
        total_violations=len(logs),
        by_exercise=grouped,
    )


@router.get("/history/{session_id}", response_model=WorkoutHistoryResponse)
def get_history(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    Get all exercise sessions (workout history) linked to a Module 1 session.
    Returns most recent sessions first.
    """
    sessions = db.query(ExerciseSession).filter(
        ExerciseSession.session_id == session_id
    ).order_by(ExerciseSession.started_at.desc()).all()

    workouts = []
    for es in sessions:
        fb = db.query(SessionFeedback).filter(
            SessionFeedback.exercise_session_id == es.id
        ).first()

        workouts.append(WorkoutHistoryItem(
            exercise_session_id=es.id,
            status=es.status,
            started_at=str(es.started_at) if es.started_at else None,
            ended_at=str(es.ended_at)     if es.ended_at   else None,
            total_reps=es.total_reps or 0,
            exercises=json.loads(es.exercises_performed) if es.exercises_performed else [],
            overall_score=fb.overall_score if fb else None,
        ))

    return WorkoutHistoryResponse(
        session_id=session_id,
        total=len(workouts),
        workouts=workouts,
    )


@router.get("/summary/{session_id}")
def get_exercise_summary(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    Aggregate stats across ALL completed workout sessions for a user.
    Shows total reps, form quality percentage, and top form issues per exercise.
    Great for viva — demonstrates progress tracking over time.
    """
    ex_sessions = db.query(ExerciseSession).filter(
        ExerciseSession.session_id == session_id,
        ExerciseSession.status     == "completed",
    ).all()

    if not ex_sessions:
        return {
            "session_id": session_id,
            "message":    "No completed workout sessions found for this user.",
            "summary":    {},
        }

    ex_ids = [es.id for es in ex_sessions]

    all_reps  = db.query(RepLog).filter(RepLog.exercise_session_id.in_(ex_ids)).all()
    all_viols = db.query(FormFeedbackLog).filter(
        FormFeedbackLog.exercise_session_id.in_(ex_ids)
    ).all()

    exercises = list({r.exercise_name for r in all_reps})
    summary   = {}

    for ex in exercises:
        ex_reps  = [r for r in all_reps  if r.exercise_name == ex]
        ex_viols = [v for v in all_viols if v.exercise_name == ex]

        good_pct = (
            sum(1 for r in ex_reps if r.rep_quality == "good") / len(ex_reps)
            if ex_reps else 0
        )

        # Count most common violations
        viol_counts: dict = {}
        for v in ex_viols:
            viol_counts[v.violation_type] = viol_counts.get(v.violation_type, 0) + 1

        top_issues = sorted(viol_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        summary[ex] = {
            "total_reps":    len(ex_reps),
            "good_form_pct": round(good_pct * 100, 1),
            "top_issues": [
                {"issue": k, "occurrences": v}
                for k, v in top_issues
            ],
        }

    return {
        "session_id":       session_id,
        "total_workouts":   len(ex_sessions),
        "exercise_summary": summary,
    }