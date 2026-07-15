"""
migrate_module2.py
===================
Run this ONCE to add the 4 new Module 2 tables to your existing database.

Safe to run multiple times — skips tables that already exist.
Does NOT touch or modify any existing Module 1 tables.

Usage:
    python migrate_module2.py
"""

import sys
from sqlalchemy import inspect
from database import engine, Base

# Import models so Base.metadata knows about all tables
from models import (
    ExerciseSession,
    RepLog,
    FormFeedbackLog,
    SessionFeedback,
)

NEW_TABLES = [
    "exercise_sessions",
    "rep_logs",
    "form_feedback_logs",
    "session_feedback",
]


def run():
    inspector = inspect(engine)
    existing  = inspector.get_table_names()

    print("\n" + "=" * 55)
    print("  BioFitCoach — Module 2 Database Migration")
    print("=" * 55)
    print(f"  Database : {engine.url}")
    print(f"  Existing tables : {existing}\n")

    to_create  = [t for t in NEW_TABLES if t not in existing]
    already_ok = [t for t in NEW_TABLES if t in existing]

    for t in already_ok:
        print(f"  ⏭  SKIP    '{t}' already exists")

    if not to_create:
        print("\n  ✅ All Module 2 tables already exist. Nothing to do.\n")
        return

    tables_objs = [
        Base.metadata.tables[t]
        for t in to_create
        if t in Base.metadata.tables
    ]
    Base.metadata.create_all(bind=engine, tables=tables_objs)

    print()
    for t in to_create:
        print(f"  ✅ CREATED '{t}'")

    print(f"\n  Done! {len(to_create)} table(s) created.")
    print("  Start the API: uvicorn main:app --reload")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run()