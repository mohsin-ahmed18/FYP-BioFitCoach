-- Run this once against the database named in DATABASE_URL (pgAdmin: Query Tool on that DB).
-- Creates all tables expected by the FastAPI routers.

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    capture_time TIME NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    elbow_angle DOUBLE PRECISION NOT NULL,
    hip_angle DOUBLE PRECISION NOT NULL,
    knee_angle DOUBLE PRECISION NOT NULL,
    shoulder_width DOUBLE PRECISION NOT NULL,
    hip_width DOUBLE PRECISION NOT NULL,
    torso_leg_ratio DOUBLE PRECISION NOT NULL,
    body_angle DOUBLE PRECISION NOT NULL,
    posture_label VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS biomechanical_profiles (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    alignment_score DOUBLE PRECISION NOT NULL,
    stability_score DOUBLE PRECISION NOT NULL,
    symmetry_score DOUBLE PRECISION NOT NULL,
    biomechanical_readiness_index DOUBLE PRECISION NOT NULL,
    structural_bias TEXT NOT NULL,
    primary_limit_factor TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workout_plans (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    llm_output TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_frames_session_id ON frames (session_id);
CREATE INDEX IF NOT EXISTS idx_biomechanical_profiles_session_id ON biomechanical_profiles (session_id);
CREATE INDEX IF NOT EXISTS idx_workout_plans_session_id ON workout_plans (session_id);
