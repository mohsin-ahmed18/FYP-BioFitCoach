"""
services/voice_agent.py — Real-Time Voice Feedback Agent
=========================================================
Provides spoken form correction and rep announcements during live workouts.

Design decisions:
  - Per-violation cooldown timers prevent the same message repeating every frame
  - Priority queue speaks errors before warnings (injury prevention first)
  - One message spoken per frame cycle maximum (no overlapping speech)
  - All TTS runs in daemon threads — never blocks the frame processing loop
  - Thread lock ensures only one pyttsx3 instance speaks at a time
    (pyttsx3 is NOT thread-safe — the lock is critical)

Vocabulary:
  - pyttsx3       : Python text-to-speech library that uses your OS's built-in
                    TTS engine (SAPI5 on Windows, NSSpeechSynthesizer on macOS,
                    espeak on Linux). Works offline, zero cost.
  - Cooldown timer: After a violation is spoken, that same violation is silenced
                    for N seconds to avoid repetitive nagging.
  - Daemon thread : A background thread that dies automatically when the main
                    program exits. Safe for fire-and-forget TTS calls.
  - Priority queue: Errors are spoken before warnings — a "back rounding" error
                    (injury risk) always beats a "go deeper" warning.

Place this file at: services/voice_agent.py
"""

import time
import threading
import logging
from collections import defaultdict
from typing import List, Optional

import pyttsx3

logger = logging.getLogger("biofitcoach.voice")


# ── Cooldown configuration ────────────────────────────────────────────────
# How long (seconds) to wait before repeating the same violation message.
# Tune these for your demo — shorter = more frequent feedback.

COOLDOWN_ERROR   = 4.0   # errors repeat every 4s  (urgent — injury risk)
COOLDOWN_WARNING = 6.5   # warnings repeat every 6.5s (helpful but not urgent)
COOLDOWN_REP     = 0.0   # rep announcements: no cooldown (once per rep by design)
COOLDOWN_IDLE    = 10.0  # motivational prompts every 10s max

# ── TTS voice settings ────────────────────────────────────────────────────
SPEECH_RATE   = 160   # words per minute (140–180 is natural; 160 = clear coaching pace)
SPEECH_VOLUME = 0.95  # 0.0–1.0

# ── Rep announcement templates ────────────────────────────────────────────
REP_MESSAGES = {
    "good":       [
        "{count} reps. Great form.",
        "That's {count}. Keep it up.",
        "{count} done. Nice depth.",
        "Good rep. {count} total.",
    ],
    "acceptable": [
        "Rep {count}. Watch your form.",
        "{count} done. Stay controlled.",
        "Rep {count}. You can do better.",
    ],
    "poor": [
        "Rep {count}. Fix your technique.",
        "{count}, but focus on form.",
        "Rep {count}. Fix it next one.",
    ],
}

# ── Milestone messages (every 5 reps) ─────────────────────────────────────
MILESTONE_MESSAGES = {
    5:  "5 reps. Halfway there.",
    10: "10 reps. Excellent work.",
    15: "15 reps. You are on fire.",
    20: "20 reps. Incredible session.",
}


class VoiceAgent:
    """
    Singleton voice feedback agent.
    Instantiated once at module level and shared across all active sessions.

    Thread safety:
        - self._lock  ensures pyttsx3 is only used by one thread at a time
        - self._speaking tracks whether TTS is currently active
        - Per-session cooldown state stored in self._session_cooldowns
    """

    def __init__(self):
        self._engine   = None
        self._lock     = threading.Lock()
        self._speaking = False
        self._enabled  = True   # can be toggled off per user preference

        # Per-session cooldown: session_key → {violation_type → last_spoken_time}
        self._session_cooldowns: dict = defaultdict(lambda: defaultdict(float))

        # Rep count cache: session_key → {exercise → last_announced_rep}
        self._last_rep_announced: dict = defaultdict(lambda: defaultdict(int))

        self._init_engine()

    # ── Engine initialisation ─────────────────────────────────────────────

    def _init_engine(self):
        """
        Initialise pyttsx3. Logs a warning instead of crashing if TTS
        is unavailable (e.g. headless CI server without audio drivers).
        """
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   SPEECH_RATE)
            self._engine.setProperty("volume", SPEECH_VOLUME)

            # Optional: choose a specific voice
            # List available voices with:  [v.id for v in self._engine.getProperty('voices')]
            # voices = self._engine.getProperty("voices")
            # self._engine.setProperty("voice", voices[0].id)   # 0=male, 1=female (OS-dependent)

            logger.info("VoiceAgent: pyttsx3 TTS engine initialised successfully")

        except Exception as e:
            logger.warning(
                f"VoiceAgent: pyttsx3 failed to initialise ({e}). "
                "Voice feedback disabled. Install pyttsx3 and audio drivers to enable."
            )
            self._engine  = None
            self._enabled = False

    # ── Public API ────────────────────────────────────────────────────────

    def enable(self):
        """Re-enable voice feedback (e.g. user turns it on mid-session)."""
        self._enabled = True
        logger.info("VoiceAgent: feedback enabled")

    def disable(self):
        """Disable all voice feedback without destroying the engine."""
        self._enabled = False
        logger.info("VoiceAgent: feedback disabled")

    def give_form_feedback(
        self,
        session_key:   str,
        violations:    List[str],
        feedback_msgs: List[str],
        severity:      str,
    ):
        """
        Speak the highest-priority form correction that has passed its cooldown.

        Called every frame from process_frame. Speaks at most ONE message
        per call — the most urgent un-cooled violation.

        Parameters
        ----------
        session_key   : str(exercise_session_id) — used for per-session cooldowns
        violations    : list of violation type strings (e.g. ['knee_cave', 'shallow_depth'])
        feedback_msgs : parallel list of human-readable messages
        severity      : 'ok' | 'warning' | 'error' — overall severity of the FormResult
        """
        if not self._enabled or not violations or not self._engine:
            return

        now = time.time()
        cooldowns = self._session_cooldowns[session_key]

        # Sort so errors come before warnings
        paired = list(zip(violations, feedback_msgs))
        if severity == "error":
            paired.sort(key=lambda x: (0 if "error" in x[0].lower() else 1))

        for violation, message in paired:
            cd = COOLDOWN_ERROR if severity == "error" else COOLDOWN_WARNING
            if now - cooldowns[violation] >= cd:
                cooldowns[violation] = now
                self._speak_async(message)
                return   # one message per frame — prevents overlapping speech

    def announce_rep(
        self,
        session_key:   str,
        exercise_name: str,
        rep_number:    int,
        rep_quality:   str,
    ):
        """
        Announce a completed repetition with quality-adaptive language.

        Only fires if rep_number is higher than the last announced rep
        for this exercise in this session — prevents duplicate announcements
        if log_rep is called more than once.

        Parameters
        ----------
        session_key   : str(exercise_session_id)
        exercise_name : e.g. 'squat'
        rep_number    : sequential rep number (1, 2, 3...)
        rep_quality   : 'good' | 'acceptable' | 'poor'
        """
        if not self._enabled or not self._engine:
            return

        last = self._last_rep_announced[session_key][exercise_name]
        if rep_number <= last:
            return   # already announced this rep

        self._last_rep_announced[session_key][exercise_name] = rep_number

        # Pick message template (cycle through list for variety)
        templates = REP_MESSAGES.get(rep_quality, REP_MESSAGES["acceptable"])
        template  = templates[(rep_number - 1) % len(templates)]
        message   = template.format(count=rep_number)

        self._speak_async(message)

        # Milestone check (5, 10, 15, 20 reps)
        if rep_number in MILESTONE_MESSAGES:
            # Delay 1.5s so milestone doesn't overlap with rep announcement
            threading.Timer(
                1.5,
                lambda: self._speak_async(MILESTONE_MESSAGES[rep_number])
            ).start()

    def motivate(self, session_key: str, rep_count: int):
        """
        Optional: call periodically to give encouragement during rest periods.
        Uses its own cooldown so it doesn't interrupt form feedback.

        Parameters
        ----------
        rep_count : total reps in session (used to pick relevant message)
        """
        if not self._enabled or not self._engine:
            return

        now      = time.time()
        cooldown = self._session_cooldowns[session_key]
        key      = "__idle__"

        if now - cooldown[key] < COOLDOWN_IDLE:
            return

        cooldown[key] = now

        if rep_count == 0:
            msg = "Ready when you are. Begin your set."
        elif rep_count < 5:
            msg = "Keep going. Good start."
        elif rep_count < 15:
            msg = "Strong work. Maintain your form."
        else:
            msg = "Outstanding session. Finish strong."

        self._speak_async(msg)

    def clear_session(self, session_key: str):
        """
        Clean up session state when a workout ends.
        Call this from end_session in the router.
        """
        self._session_cooldowns.pop(session_key, None)
        self._last_rep_announced.pop(session_key, None)
        logger.info(f"VoiceAgent: cleared session state for {session_key}")

    def speak(self, text: str):
        """
        Speak any arbitrary message immediately (bypasses cooldowns).
        Use for start/end announcements, errors, or test calls.
        """
        if self._enabled and self._engine:
            self._speak_async(text)

    # ── Internal TTS execution ────────────────────────────────────────────

    def _speak_async(self, text: str):
        """
        Speak text in a background daemon thread.

        The threading.Lock() ensures only one runAndWait() call executes
        at a time — pyttsx3 crashes if called concurrently from multiple
        threads without serialisation.

        If the engine is already speaking (lock held), the new message
        is dropped rather than queued — this prevents a backlog of stale
        form corrections playing minutes after they were relevant.
        """
        if not text or not self._engine:
            return

        def _run():
            # Try to acquire lock — skip if TTS is already busy
            acquired = self._lock.acquire(blocking=False)
            if not acquired:
                logger.debug(f"VoiceAgent: skipped '{text[:30]}...' — engine busy")
                return
            try:
                self._speaking = True
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.warning(f"VoiceAgent: TTS error — {e}")
            finally:
                self._speaking  = False
                self._lock.release()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def __del__(self):
        """Clean up pyttsx3 engine on garbage collection."""
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass


# ── Module-level singleton ────────────────────────────────────────────────
# Loaded once when the module is first imported.
# All routers share this single instance.
voice_agent = VoiceAgent()