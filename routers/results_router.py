from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db
from services.reps_service import count_reps_from_angle_series, count_reps_from_knee_angles

router = APIRouter(tags=["Results"])


@router.get("/results/{session_id}")
def get_results(session_id: int, db: Session = Depends(get_db)):

    result = db.execute(
        text("""
            SELECT *
            FROM biomechanical_profiles
            WHERE session_id = :sid
        """),
        {"sid": session_id}
    ).fetchone()

    if not result:
        return {"error": "No analysis found"}

    return dict(result._mapping)


@router.get("/reps/{session_id}")
def get_reps(session_id: int, db: Session = Depends(get_db)):
    """
    Rep counting endpoint.

    Uses the stored `frames.knee_angle` series for the session and applies a simple peak/valley detector.
    """
    session = db.execute(
        text("SELECT id FROM sessions WHERE id = :sid"),
        {"sid": session_id},
    ).fetchone()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rows = db.execute(
        text(
            """
            SELECT elbow_angle, knee_angle
            FROM frames
            WHERE session_id = :sid
            ORDER BY id ASC
            """
        ),
        {"sid": session_id},
    ).fetchall()

    elbow_angles = [float(r[0]) for r in rows if r and r[0] is not None]
    knee_angles = [float(r[1]) for r in rows if r and r[1] is not None]

    # For barbell curls, elbow angle is the best oscillator (extended ~170 -> flexed ~50 -> extended).
    # We compute both and choose the signal that yields more reps.
    reps_elbow = count_reps_from_angle_series(
        elbow_angles,
        fps=10.0,
        # Curls have bigger ROM but more noise/occlusion; be more permissive.
        min_drop_deg=20.0,
        min_rep_frames=3,
        mode="curl",
    )
    reps_knee = count_reps_from_knee_angles(knee_angles, fps=10.0)

    def _knee_series_unusable_for_squats(
        k: list[float],
        rk: list,
        re: list,
    ) -> bool:
        """
        When hips/legs are off-camera, MediaPipe still outputs knee angles that jitter.
        That yields bogus squat-style reps. Prefer elbow-driven counting when the knee
        series never shows real flexion depth, is almost flat, or matches the client's
        upper-body-only placeholder (~135° flatline).
        """
        if len(k) < 12:
            return False
        lo, hi = min(k), max(k)
        rom = hi - lo
        mean_v = sum(k) / len(k)
        # Flat placeholder or near-constant noise
        if rom < 18.0:
            return True
        if 132.0 <= mean_v <= 138.0 and rom < 12.0:
            return True
        # No meaningful squat depth (standing-ish knee angle throughout)
        if lo > 96.0 and rom < 42.0:
            return True
        # Jittery mid-range knee signal producing many bogus squat reps vs few elbow reps
        if lo > 72.0 and rom < 55.0 and len(rk) > len(re) + 1 and len(re) >= 1:
            return True
        return False

    if _knee_series_unusable_for_squats(knee_angles, reps_knee, reps_elbow):
        reps = reps_elbow
        source = "elbow_angle"
    elif len(reps_elbow) >= len(reps_knee):
        reps = reps_elbow
        source = "elbow_angle"
    else:
        reps = reps_knee
        source = "knee_angle"

    return {
        "session_id": session_id,
        "source": source,
        "rep_count": len(reps),
        "reps": [
            {
                "rep_number": r.rep_number,
                "quality": r.quality,
                "feedback": r.feedback,
                "tempo": r.tempo,
                "depth": r.depth,
                "stability": r.stability,
            }
            for r in reps
        ],
    }