from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from database import engine
import json
from services.analysis_service import run_biomechanical_analysis
from services.recommendation_service import rule_based_recommendations
from services.llm_service import generate_ai_workout

router = APIRouter()


@router.post("/generate-workout")
def generate_workout(
    session_id: int,
    exercise: str | None = Query(
        None,
        description="Exercise key from the app (e.g. squat, bench-press) for LLM context",
    ),
):

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    with engine.connect() as conn:

        # -----------------------------
        # 1️⃣ Get session frames
        # -----------------------------
        session_frames = conn.execute(
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
                WHERE session_id = :sid
            """),
            {"sid": session_id}
        ).mappings().all()

        if not session_frames:
            raise HTTPException(status_code=404, detail="No frames found")

        session_frames = [dict(row) for row in session_frames]

        # -----------------------------
        # 2️⃣ Get population frames
        # -----------------------------
        population_frames = conn.execute(
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
        ).mappings().all()

        population_frames = [dict(row) for row in population_frames]

    # -----------------------------
    # 3️⃣ Analysis
    # -----------------------------
    readiness_vector = run_biomechanical_analysis(
        session_frames,
        population_frames
    )

    # -----------------------------
    # 4️⃣ Exercise recommendation
    # -----------------------------
    exercises = rule_based_recommendations(readiness_vector)

    # -----------------------------
    # 5️⃣ LLM generation
    # -----------------------------
    ai_report = generate_ai_workout(readiness_vector, exercises, session_exercise=exercise)

    # Persist short rule-based exercise names separately so the web UI can show compact
    # chips; the LLM "recommendations" entries are often long prose sentences.
    plan_payload = {**ai_report, "rule_based_exercise_tags": exercises}

    # -----------------------------
    # 6️⃣ SAVE TO DATABASE ✅
    # -----------------------------
    with engine.begin() as conn:  # auto commit

        conn.execute(
            text("""
                INSERT INTO workout_plans (session_id, llm_output)
                VALUES (:sid, :output)
            """),
            {
                "sid": session_id,
                "output": json.dumps(plan_payload)
            }
        )

    # -----------------------------
    # 7️⃣ Return response
    # -----------------------------
    return {
        "session_id": session_id,
        "readiness_vector": readiness_vector,
        "recommended_exercises": exercises,
        "ai_report": ai_report
    }

@router.get("/workout-plan/{session_id}")
def get_workout_plan(session_id: int):

    with engine.connect() as conn:

        result = conn.execute(
            text("""
                SELECT id, session_id, llm_output, created_at
                FROM workout_plans
                WHERE session_id = :sid
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"sid": session_id}
        ).mappings().first()

        if not result:
            raise HTTPException(status_code=404, detail="No workout plan found")

        return dict(result)