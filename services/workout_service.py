from sqlalchemy import text
from services.recommendation_service import rule_based_recommendations


def generate_workout(session_id, db):

    profile = db.execute(text("""
        SELECT *
        FROM biomechanical_profiles
        WHERE session_id = :sid
    """), {"sid": session_id}).fetchone()

    if not profile:
        return {"error": "analysis not found"}

    profile = dict(profile._mapping)

    exercises = rule_based_recommendations(profile)

    for exercise in exercises:

        db.execute(text("""
            INSERT INTO workout_plans (session_id, exercise_name)
            VALUES (:sid,:ex)
        """), {
            "sid": session_id,
            "ex": exercise
        })

    db.commit()

    return {
        "recommended_exercises": exercises
    }