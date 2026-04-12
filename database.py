"""
PostgreSQL connection for SQLAlchemy.

Configuration
-------------
Set DATABASE_URL in a `.env` file next to `main.py` (see `.env.example`).
Example:
    DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@127.0.0.1:5432/Biomechanics_AI

CLI usage
---------
From this project folder (with venv activated):

    python database.py              # verify DB is reachable, then exit 0 or 1
    python database.py init         # create tables (sql/init_schema.sql) — run once
    python database.py serve        # verify DB, then start uvicorn (API on :8000)

PostgreSQL itself must already be running as a Windows service; this code only checks auth + TCP.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()


def _normalize_database_url(url: str) -> str:
    """Prefer IPv4 loopback so libpq does not default to ::1 on Windows."""
    return url.replace("@localhost:", "@127.0.0.1:").replace("@localhost/", "@127.0.0.1/")


_raw_url = os.getenv("DATABASE_URL", "").strip()
if not _raw_url:
    raise RuntimeError(
        "DATABASE_URL is not set. Copy .env.example to .env and set your connection string."
    )

DATABASE_URL = _normalize_database_url(_raw_url)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_database_connection() -> None:
    """Run a trivial query. Raises if the server is down or password/database is wrong."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def _safe_url_hint(url: str) -> str:
    """Log host/db without password (best-effort)."""
    try:
        if "@" in url:
            return url.split("@", 1)[-1]
    except Exception:
        pass
    return "(could not parse URL)"


def _cli_check() -> int:
    try:
        verify_database_connection()
        print("Database OK:", _safe_url_hint(DATABASE_URL))
        return 0
    except Exception as e:
        print("Database check FAILED:", e, file=sys.stderr)
        print(
            "Fix DATABASE_URL in .env (same password as pgAdmin for user postgres).",
            file=sys.stderr,
        )
        return 1


def _cli_serve() -> None:
    verify_database_connection()
    print("Database OK:", _safe_url_hint(DATABASE_URL))
    print("Starting API at http://127.0.0.1:8000 (Ctrl+C to stop)")
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


def _cli_init() -> int:
    """Apply sql/init_schema.sql (idempotent CREATE IF NOT EXISTS)."""
    from pathlib import Path

    sql_path = Path(__file__).resolve().parent / "sql" / "init_schema.sql"
    if not sql_path.is_file():
        print("Missing schema file:", sql_path, file=sys.stderr)
        return 1
    raw = sql_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in raw.split(";")]

    def _strip_comment_lines(block: str) -> str:
        lines = [ln for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("--")]
        return "\n".join(lines).strip()

    statements = [_strip_comment_lines(c) for c in chunks]
    statements = [s for s in statements if s]
    try:
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
    except Exception as e:
        print("Schema init failed:", e, file=sys.stderr)
        return 1
    print("Schema OK (tables ready):", sql_path.name, "on", _safe_url_hint(DATABASE_URL))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        sys.exit(_cli_check())
    if cmd == "init":
        sys.exit(_cli_init())
    if cmd == "serve":
        try:
            _cli_serve()
        except Exception as e:
            print("Cannot start server:", e, file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: python database.py [check|init|serve]", file=sys.stderr)
        sys.exit(2)
