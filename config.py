"""
config.py
==========
Central settings file. Reads every value from your .env file.
All other files import `settings` from here — never read os.environ directly.

Add these to your .env file:
    DATABASE_URL=postgresql+psycopg2://postgres:yourpassword@localhost:5432/Biomechanics_AI
    OPENAI_API_KEY=sk-...
    GRU_MODEL_PATH=ml_models/gru_model.keras
    SCALER_PATH=ml_models/scaler.pkl
    LABEL_MAPPING_PATH=ml_models/label_mapping.json
    SEQUENCE_LENGTH=30
    CONFIDENCE_THRESHOLD=0.70
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── Database ────────────────────────────────────────────────────────────
    database_url: str

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str

    # ── Module 2 — ML model paths ────────────────────────────────────────────
    gru_model_path:       str   = "ml_models/gru_model.keras"
    scaler_path:          str   = "ml_models/scaler.pkl"
    label_mapping_path:   str   = "ml_models/label_mapping.json"
    model_config_path:    str   = "ml_models/model_config.json"

    # ── Module 2 — Inference ─────────────────────────────────────────────────
    confidence_threshold: float = 0.70
    sequence_length:      int   = 30    # must match training notebook value

    # ── App ──────────────────────────────────────────────────────────────────
    app_title:   str = "BioFitCoach API"
    app_version: str = "2.0.0"

    class Config:
        env_file = ".env"
        extra    = "ignore"


settings = Settings()