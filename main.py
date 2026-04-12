from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from database import engine
from routers import (
    session_router,
    frames_router,
    analysis_router,
    results_router,
    workout_router,
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK")
    except Exception as e:
        logger.error("Database connection failed: %s", e)
        logger.error(
            "Fix DATABASE_URL in FYP-BioFitCoach/.env — use the exact password for user "
            '"postgres" (pgAdmin: right-click server → Properties → Connection). '
            "Special characters in the password must be URL-encoded in DATABASE_URL."
        )
    yield


app = FastAPI(title="Biomechanics Analysis ", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    # Next dev on LAN (e.g. http://10.x.x.x:3000) when calling the API directly
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|[0-9]{1,3}(\.[0-9]{1,3}){3}):3000$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router.router)
app.include_router(frames_router.router)
app.include_router(analysis_router.router)
app.include_router(results_router.router)
app.include_router(workout_router.router)