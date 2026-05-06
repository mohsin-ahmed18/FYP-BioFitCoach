"""
main.py — BioFitCoach Unified FastAPI Application
==================================================
Mounts ALL routers from both Module 1 and Module 2 into one app.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

API Docs:
    http://localhost:8000/docs     ← Swagger UI (interactive)
    http://localhost:8000/redoc    ← ReDoc (clean docs)
    http://localhost:8000/health   ← quick health check

Module 1 endpoints (existing):
    POST /session                  Create user + body-scan session
    POST /frames                   Upload pose frames
    POST /analysis                 Run biomechanical analysis
    GET  /results/{session_id}     Get analysis results
    POST /generate-workout         Generate AI workout plan

Module 2 endpoints (new):
    POST /exercise/start-session   Link to Module 1 session, start workout
    POST /exercise/process-frame   Send webcam frames, get live detection
    POST /exercise/log-rep         Save a rep to database
    POST /exercise/end-session     End session, generate AI coaching report
    GET  /exercise/session/{id}    Get session summary
    GET  /exercise/session/{id}/reps  Get all rep logs
    GET  /feedback/session/{id}    Get LLM coaching report
    GET  /feedback/session/{id}/violations  Form violation history
    GET  /feedback/history/{session_id}     Workout history for a user
    GET  /feedback/summary/{session_id}     Aggregate stats across sessions
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from config import settings

# ── Module 1 routers (existing) ───────────────────────────────────────────
from routers.session_router  import router as session_router
from routers.frames_router   import router as frames_router
from routers.analysis_router import router as analysis_router
from routers.results_router  import router as results_router
from routers.workout_router  import router as workout_router

# ── Module 2 routers (new) ────────────────────────────────────────────────
from routers.exercise_session_router import router as exercise_router
from routers.rep_feedback_router     import router as feedback_router

# ── Database (for table creation on startup) ──────────────────────────────
from database import engine
from models import Base    # Module 2 ORM models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("biofitcoach")


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version} — Starting Up")
    logger.info("=" * 60)

    # Create Module 2 tables if they don't exist yet
    # Safe to run every startup — skips existing tables
    Base.metadata.create_all(bind=engine)
    logger.info("  ✅ Module 2 DB tables verified")

    # BiLSTM model is loaded as a singleton in exercise_detector.py
    # It prints its own status (loaded / pose-only) during import
    from services.exercise_detector import detector
    status = "loaded" if detector.model_loaded else "POSE-ONLY (model not found)"
    logger.info(f"  ✅ BiLSTM detector : {status}")
    logger.info(f"  ✅ API ready       : http://localhost:8000/docs")
    logger.info("=" * 60)

    yield  # ← server is live and serving requests

    # ── SHUTDOWN ───────────────────────────────────────────────────────
    logger.info("  BioFitCoach shutting down...")


# ── App creation ──────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## BioFitCoach API — AI-Powered Personal Trainer

### Module 1 — Body Analysis
Analyses body proportions and biomechanics from pose frames.
Generates personalised AI workout plans using OpenAI.

### Module 2 — Exercise Coach
Real-time exercise classification (22 exercises), rep counting,
biomechanical form checking, and AI post-session coaching feedback.

---
**Typical flow:**
1. `POST /session` → create user + get `session_id`
2. `POST /frames` → upload body-scan frames
3. `POST /analysis` → run biomechanical analysis
4. `POST /generate-workout` → get AI workout plan
5. `POST /exercise/start-session` → start live workout (uses same `session_id`)
6. `POST /exercise/process-frame` → stream webcam frames for detection
7. `POST /exercise/log-rep` → save each rep
8. `POST /exercise/end-session` → get AI coaching feedback
""",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code, and duration for every request."""
    start    = time.time()
    response = await call_next(request)
    ms       = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.1f}ms)")
    return response


# ── Exception handlers ────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error":   "Validation error — check your request body",
            "details": exc.errors(),
        },
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Mount Module 1 routers ────────────────────────────────────────────────
app.include_router(session_router)
app.include_router(frames_router)
app.include_router(analysis_router)
app.include_router(results_router)
app.include_router(workout_router)

# ── Mount Module 2 routers ────────────────────────────────────────────────
app.include_router(exercise_router)
app.include_router(feedback_router)


# ── Root + Health endpoints ───────────────────────────────────────────────
@app.get("/", tags=["health"], include_in_schema=False)
def root():
    return {
        "app":     settings.app_name,
        "version": settings.app_version,
        "docs":    "http://localhost:8000/docs",
    }


@app.get("/health", tags=["health"])
def health():
    """Quick health check — confirms API is running and model status."""
    from services.exercise_detector import detector
    return {
        "status":          "healthy",
        "app":             settings.app_name,
        "version":         settings.app_version,
        "model_loaded":    detector.model_loaded,
        "active_sessions": detector.active_sessions(),
        "model_path":      settings.gru_model_path,
    }