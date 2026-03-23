from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db

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