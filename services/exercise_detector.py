"""
services/exercise_detector.py — BiLSTM Exercise Detection Engine
=================================================================
Loads the trained BiLSTM model once at startup and processes
incoming webcam frames to classify exercises in real time.

Key design decisions:
  - Model loaded ONCE as a module-level singleton (expensive to load).
  - Per-session frame buffers stored in a dict keyed by exercise_session_id.
  - EMA (Exponential Moving Average) smoothing reduces landmark jitter.
  - Falls back to POSE-ONLY mode if model files are missing — so you can
    test the API and database pipeline before the model is ready.

Vocabulary:
- BiLSTM    : Bidirectional LSTM — processes the 30-frame sequence both
              forward and backward for richer temporal context.
- Frame buffer: Rolling deque of the last 30 feature vectors. When full,
              the model makes a prediction and the oldest frame drops off.
- EMA       : Exponential Moving Average — smooths noisy landmark positions.
              new_value = (1 - α) × old_smooth + α × new_raw   (α = 0.3)
- Confidence: The softmax probability assigned to the top-1 class [0–1].
              Below CONFIDENCE_THRESHOLD → "No Exercise Detected".
"""

import cv2
import json
import joblib
import logging
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

import mediapipe as mp

from services.feature_engineering import (
    extract_frame_features,
    extract_rule_angles,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
)
from config import settings

logger = logging.getLogger("biofitcoach")


class ExerciseDetector:
    """
    Singleton inference engine shared across all active sessions.
    Instantiated once when the module is first imported (app startup).
    """

    def __init__(self):
        self.model              = None
        self.scaler             = None
        self.label_mapping:     Dict[int, str] = {}
        self.model_loaded:      bool = False

        # MediaPipe pose estimator (shared — stateless per frame)
        mp_pose    = mp.solutions.pose
        self.pose  = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
        )

        # Per-session state: session_key → deque of feature vectors
        self._buffers: Dict[str, deque] = {}
        # Per-session EMA state: session_key → list of smoothed landmark dicts
        self._ema:     Dict[str, Optional[list]] = {}

        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self):
        """
        Load BiLSTM model + scaler + label map.
        If any file is missing, run in POSE-ONLY mode (no classification).
        This lets you test the full API pipeline before training finishes.
        """
        import os
        try:
            import tensorflow as tf

            if not os.path.exists(settings.gru_model_path):
                logger.warning(
                    f"Model not found at '{settings.gru_model_path}'. "
                    "Running in POSE-ONLY mode — exercise classification disabled. "
                    "Copy your trained model to ml_models/ to enable it."
                )
                return

            logger.info(f"Loading BiLSTM model from {settings.gru_model_path} ...")
            self.model  = tf.keras.models.load_model(settings.gru_model_path)
            self.scaler = joblib.load(settings.scaler_path)

            with open(settings.label_mapping_path, "r") as f:
                raw = json.load(f)
            self.label_mapping = {int(k): v for k, v in raw.items()}

            self.model_loaded = True
            num_classes       = len(self.label_mapping)
            logger.info(f"BiLSTM loaded — {num_classes} exercise classes: {list(self.label_mapping.values())}")

        except Exception as e:
            logger.warning(f"Model load failed ({e}). Running in POSE-ONLY mode.")

    # ── Session buffer management ──────────────────────────────────────────

    def init_session(self, session_key: str):
        """Create a fresh frame buffer for a new exercise session."""
        self._buffers[session_key] = deque(maxlen=SEQUENCE_LENGTH)
        self._ema[session_key]     = None

    def clear_session(self, session_key: str):
        """Remove frame buffer when session ends (free memory)."""
        self._buffers.pop(session_key, None)
        self._ema.pop(session_key, None)

    def active_sessions(self) -> int:
        return len(self._buffers)

    # ── Core per-frame inference ───────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray, session_key: str) -> Dict:
        """
        Process one BGR video frame and return a detection result dict.

        Steps:
            1. MediaPipe pose detection
            2. EMA landmark smoothing
            3. Extract 48-feature vector
            4. Append to rolling frame buffer
            5. Run BiLSTM when buffer is full (30 frames)
            6. Return results

        Returns
        -------
        dict with keys:
            person_detected, exercise, confidence,
            rule_angles, buffer_ready, buffer_status
        """
        if session_key not in self._buffers:
            self.init_session(session_key)

        # 1. MediaPipe detection
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            self._buffers[session_key].clear()
            self._ema[session_key] = None
            return self._empty_result()

        # 2. EMA smoothing
        self._apply_ema(session_key, results.pose_landmarks)

        # 3. Feature extraction
        features    = extract_frame_features(results.pose_landmarks)
        rule_angles = extract_rule_angles(results.pose_landmarks)

        if features is None:
            return self._empty_result()

        # 4. Update buffer
        buf = self._buffers[session_key]
        buf.append(features)
        buffer_ready  = len(buf) == SEQUENCE_LENGTH
        buffer_status = f"{len(buf)}/{SEQUENCE_LENGTH}"

        # 5. Predict
        exercise   = "Warming up..." if not buffer_ready else "No Exercise Detected"
        confidence = 0.0

        if buffer_ready:
            if self.model_loaded:
                exercise, confidence = self._predict(buf)
            else:
                exercise = "Model Not Loaded"

        return {
            "person_detected": True,
            "exercise":        exercise,
            "confidence":      round(float(confidence), 3),
            "rule_angles":     rule_angles,
            "buffer_ready":    buffer_ready,
            "buffer_status":   buffer_status,
        }

    def _predict(self, buffer: deque) -> Tuple[str, float]:
        """Run the BiLSTM on a full 30-frame buffer."""
        sequence      = np.array(list(buffer), dtype=np.float32)     # (30, 48)
        scaled        = self.scaler.transform(sequence)               # StandardScaler
        input_tensor  = scaled[np.newaxis, ...]                       # (1, 30, 48)

        probs         = self.model.predict(input_tensor, verbose=0)[0]
        pred_idx      = int(np.argmax(probs))
        confidence    = float(probs[pred_idx])

        if confidence < settings.confidence_threshold:
            return "No Exercise Detected", confidence

        exercise_name = self.label_mapping.get(pred_idx, "Unknown")
        return exercise_name, confidence

    def _apply_ema(self, session_key: str, landmarks, alpha: float = 0.3):
        """
        Apply EMA smoothing to landmark coordinates in-place.
        Reduces per-frame jitter from MediaPipe on fast movements.
        α=0.3 → 30% new measurement, 70% smoothed history.
        """
        raw = landmarks.landmark
        if self._ema[session_key] is None:
            self._ema[session_key] = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in raw
            ]
        else:
            for i, lm in enumerate(raw):
                s               = self._ema[session_key][i]
                s["x"]          = (1 - alpha) * s["x"]         + alpha * lm.x
                s["y"]          = (1 - alpha) * s["y"]         + alpha * lm.y
                s["z"]          = (1 - alpha) * s["z"]         + alpha * lm.z
                s["visibility"] = (1 - alpha) * s["visibility"] + alpha * lm.visibility

            for i, lm in enumerate(raw):
                s               = self._ema[session_key][i]
                lm.x, lm.y, lm.z, lm.visibility = s["x"], s["y"], s["z"], s["visibility"]

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "person_detected": False,
            "exercise":        "No Detection",
            "confidence":      0.0,
            "rule_angles":     {},
            "buffer_ready":    False,
            "buffer_status":   f"0/{SEQUENCE_LENGTH}",
        }

    def __del__(self):
        if hasattr(self, "pose"):
            self.pose.close()


# ── Module-level singleton ────────────────────────────────────────────────
# Loaded once at import time (when main.py starts the API).
# All routers share this single instance.
detector = ExerciseDetector()