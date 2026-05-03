"""
database.py
============
SQLAlchemy engine + session factory shared by the entire app.
Both Module 1 and Module 2 routers use get_db() from here.

CHANGED from original: DATABASE_URL now comes from config.py
so your password is in .env and never committed to git.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

DATABASE_URL = settings.database_url

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base is defined here so models.py can import it
Base = declarative_base()


def get_db():
    """
    FastAPI dependency — yields a DB session for one request, closes it after.
    Usage in any router:
        from database import get_db
        def my_route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()