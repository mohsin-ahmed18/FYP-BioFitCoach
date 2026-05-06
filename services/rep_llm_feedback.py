"""
services/rep_llm_feedback.py — Post-Session LLM Coaching Feedback
==================================================================
Called ONCE when the user ends their exercise session.

Sends the session's rep data, form violations, and (if available) the
user's Module 1 biomechanical profile to OpenAI GPT-4o-mini.

Returns a structured JSON coaching report that is saved in session_feedback.
"""

import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI
from config import settings

logger = logging.getLogger("biofitcoach")
client = OpenAI(api_key=settings.openai_api_key)


def _quality_label(score: float) -> str:
    if score >= 0.85: return "excellent"
    if score >= 0.70: return "good"
    if score >= 0.50: return "moderate"
    return "needs improvement"


def generate_session_feedback(
    rep_summary:     List[Dict],
    form_violations: List[Dict],
    bio_profile:     Optional[Dict] = None,
    workout_plan:    Optional[str]  = None,
) -> Dict:
    """
    Generate post-session coaching report using OpenAI.

    Parameters
    ----------
    rep_summary     : [{ exercise, total_reps, good_reps, poor_reps, avg_confidence }]
    form_violations : [{ exercise, violation_type, feedback_message, count }]
    bio_profile     : dict from Module 1 biomechanical_profiles table (optional)
    workout_plan    : LLM output string from Module 1 workout_plans (optional)

    Returns
    -------
    dict with keys: overall_summary, overall_score, exercise_breakdown,
                    key_issues, improvements, next_session_tips
    """

    # Build rep context block
    rep_lines = ["Workout Session Data:"]
    for ex in rep_summary:
        rep_lines.append(
            f"  - {ex['exercise']}: {ex['total_reps']} reps | "
            f"{ex['good_reps']} good form | {ex['poor_reps']} poor form | "
            f"avg model confidence {ex['avg_confidence']:.0%}"
        )

    # Build violation context block
    viol_lines = []
    if form_violations:
        viol_lines.append("\nForm Violations Detected:")
        for v in form_violations:
            viol_lines.append(
                f"  - [{v['exercise']}] {v['violation_type']}: "
                f"\"{v['feedback_message']}\" (occurred {v['count']}x)"
            )

    # Build biomechanical profile context (Module 1 integration)
    bio_lines = []
    if bio_profile:
        bio_lines = [
            "\nUser Biomechanical Profile (from body scan):",
            f"  Alignment Score  : {bio_profile.get('alignment_score', 'N/A')} "
            f"({_quality_label(bio_profile.get('alignment_score', 0))})",
            f"  Stability Score  : {bio_profile.get('stability_score', 'N/A')} "
            f"({_quality_label(bio_profile.get('stability_score', 0))})",
            f"  Symmetry Score   : {bio_profile.get('symmetry_score', 'N/A')} "
            f"({_quality_label(bio_profile.get('symmetry_score', 0))})",
            f"  Structural Bias  : {bio_profile.get('structural_bias', 'N/A')}",
            f"  Primary Limit    : {bio_profile.get('primary_limit_factor', 'N/A')}",
        ]

    plan_lines = []
    if workout_plan:
        plan_lines = [f"\nRecommended Plan from Body Analysis:\n{workout_plan[:500]}"]

    full_context = "\n".join(rep_lines + viol_lines + bio_lines + plan_lines)

    system_prompt = (
        "You are BioFitCoach, an expert AI personal trainer with deep knowledge of "
        "biomechanics and exercise science. Provide specific, actionable, and encouraging "
        "coaching feedback based on the actual workout data provided. "
        "Be honest about form issues but motivating in tone. "
        "Respond in valid JSON only — no markdown, no extra text."
    )

    user_prompt = f"""{full_context}

Respond with this exact JSON structure:
{{
  "overall_summary": "2-3 sentence summary of the overall session quality",
  "overall_score": <integer 0-100 representing overall form quality>,
  "exercise_breakdown": {{
    "<exercise_name>": {{
      "summary": "1-2 sentences about this exercise specifically",
      "strengths": ["specific strength 1", "specific strength 2"],
      "corrections": ["specific correction 1", "specific correction 2"]
    }}
  }},
  "key_issues": ["most important form issue 1", "issue 2", "issue 3"],
  "improvements": ["specific drill or cue to fix issue 1", "improvement 2"],
  "next_session_tips": "1-2 sentences of advice for the next workout"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=1200,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1][4:] if parts[1].startswith("json") else parts[1]

        return json.loads(raw.strip())

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return _fallback(rep_summary)

    except Exception as e:
        logger.error(f"OpenAI API error in generate_session_feedback: {e}")
        return _fallback(rep_summary)


def _fallback(rep_summary: List[Dict]) -> Dict:
    exercises = [ex.get("exercise", "exercise") for ex in rep_summary]
    return {
        "overall_summary":   "Session recorded successfully. Keep up the great work!",
        "overall_score":     70,
        "exercise_breakdown": {
            ex: {"summary": "Session data recorded.", "strengths": [], "corrections": []}
            for ex in exercises
        },
        "key_issues":       [],
        "improvements":     ["Focus on controlled movement throughout each rep."],
        "next_session_tips": "Maintain consistent form and gradually increase intensity.",
    }