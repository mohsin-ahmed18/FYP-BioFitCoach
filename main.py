from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from routers import (
    session_router,
    frames_router,
    analysis_router,
    results_router,
    workout_router
)

app = FastAPI(title="Biomechanics Analysis ")

app.include_router(session_router.router)
app.include_router(frames_router.router)
app.include_router(analysis_router.router)
app.include_router(results_router.router)
app.include_router(workout_router.router)