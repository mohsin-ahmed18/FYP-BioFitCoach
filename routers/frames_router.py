from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db
from schemas import FramesUpload

router = APIRouter(tags=["Frames"])


@router.post("/frames")
def upload_frames(data: FramesUpload, db: Session = Depends(get_db)):

    session = db.execute(
        text("SELECT id FROM sessions WHERE id = :sid"),
        {"sid": data.session_id}
    ).fetchone()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for frame in data.frames:

        db.execute(
            text("""
                INSERT INTO frames (
                    session_id,
                    elbow_angle,
                    hip_angle,
                    knee_angle,
                    shoulder_width,
                    hip_width,
                    torso_leg_ratio,
                    body_angle,
                    posture_label
                )
                VALUES (
                    :sid,
                    :elbow,
                    :hip,
                    :knee,
                    :sw,
                    :hw,
                    :tlr,
                    :body,
                    :label
                )
            """),
            {
                "sid": data.session_id,
                "elbow": frame.elbow_angle,
                "hip": frame.hip_angle,
                "knee": frame.knee_angle,
                "sw": frame.shoulder_width,
                "hw": frame.hip_width,
                "tlr": frame.torso_leg_ratio,
                "body": frame.body_angle,
                "label": frame.posture_label
            }
        )

    db.commit()

    return {
        "message": "Frames uploaded successfully",
        "frames_inserted": len(data.frames)
    }