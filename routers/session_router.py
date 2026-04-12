from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, date, time
from typing import Optional

from database import get_db
from schemas import SessionCreate

router = APIRouter(tags=["Session"])


def _parse_iso_to_date(value: Optional[str], fallback: date) -> date:
    if not value or not value.strip():
        return fallback
    try:
        s = value.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s).date()
    except ValueError:
        return fallback


def _parse_iso_to_time(value: Optional[str], fallback: time) -> time:
    if not value or not value.strip():
        return fallback
    try:
        s = value.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s).time()
    except ValueError:
        return fallback


@router.post("/session")
def create_session(data: SessionCreate, db: Session = Depends(get_db)):

    now = datetime.now()

    session_date = _parse_iso_to_date(data.session_date, now.date())
    capture_time = _parse_iso_to_time(data.capture_time, now.time())

    user = db.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": data.email}
    ).fetchone()

    if user:
        user_id = user[0]

    else:
        result = db.execute(
            text("""
                INSERT INTO users (name, email)
                VALUES (:name, :email)
                RETURNING id
            """),
            {"name": data.name, "email": data.email}
        )

        user_id = result.fetchone()[0]

    session = db.execute(
        text("""
            INSERT INTO sessions (user_id, session_date, capture_time)
            VALUES (:uid, :sdate, :ctime)
            RETURNING id
        """),
        {
            "uid": user_id,
            "sdate": session_date,
            "ctime": capture_time
        }
    )

    session_id = session.fetchone()[0]

    db.commit()

    return {
        "user_id": user_id,
        "session_id": session_id
    }