from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db
from schemas import AnalysisRequest

from services.analysis_service import run_biomechanical_analysis


router = APIRouter(tags=["Analysis"])


@router.post("/analysis")
def run_analysis(data: AnalysisRequest, db: Session = Depends(get_db)):

    # -------------------------------------------------
    # 1️⃣ Check if session exists
    # -------------------------------------------------

    session = db.execute(
        text("SELECT id FROM sessions WHERE id = :sid"),
        {"sid": data.session_id}
    ).fetchone()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # -------------------------------------------------
    # 2️⃣ Fetch frames for THIS session
    # -------------------------------------------------

    session_frames = db.execute(
        text("""
        SELECT
            session_id,
            elbow_angle,
            hip_angle,
            knee_angle,
            shoulder_width,
            hip_width,
            torso_leg_ratio,
            body_angle
        FROM frames
        WHERE session_id = :sid
        """),
        {"sid": data.session_id}
    ).fetchall()

    if not session_frames:
        raise HTTPException(
            status_code=404,
            detail="No frames found for this session"
        )

    session_frames = [dict(row._mapping) for row in session_frames]

    # -------------------------------------------------
    # 3️⃣ Fetch ALL frames for population statistics
    # -------------------------------------------------

    population_frames = db.execute(
        text("""
        SELECT
            elbow_angle,
            hip_angle,
            knee_angle,
            shoulder_width,
            hip_width,
            torso_leg_ratio,
            body_angle
        FROM frames
        """)
    ).fetchall()

    population_frames = [dict(row._mapping) for row in population_frames]

    # -------------------------------------------------
    # 4️⃣ Run full biomechanical analysis
    # -------------------------------------------------

    results = run_biomechanical_analysis(
        session_frames=session_frames,
        population_frames=population_frames
    )

    # -------------------------------------------------
    # 5️⃣ Save results in biomechanical_profiles
    # -------------------------------------------------

    db.execute(
        text("""
        INSERT INTO biomechanical_profiles (
            session_id,
            alignment_score,
            stability_score,
            symmetry_score,
            biomechanical_readiness_index,
            structural_bias,
            primary_limit_factor
        )
        VALUES (
            :sid,
            :align,
            :stab,
            :sym,
            :ready,
            :bias,
            :limit
        )
        """),
        {
            "sid": data.session_id,
            "align": results["alignment_score"],
            "stab": results["stability_score"],
            "sym": results["symmetry_score"],
            "ready": results["biomechanical_readiness_index"],
            "bias": results["structural_bias"],
            "limit": results["primary_limit_factor"]
        }
    )

    db.commit()

    # -------------------------------------------------
    # 6️⃣ Return results
    # -------------------------------------------------

    return {
        "session_id": data.session_id,
        "alignment_score": results["alignment_score"],
        "stability_score": results["stability_score"],
        "symmetry_score": results["symmetry_score"],
        "biomechanical_readiness_index": results["biomechanical_readiness_index"],
        "structural_bias": results["structural_bias"],
        "primary_limit_factor": results["primary_limit_factor"]
    }