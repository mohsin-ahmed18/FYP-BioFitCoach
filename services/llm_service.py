import os
import json
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_ai_workout(readiness_vector, exercises):

    alignment = readiness_vector["alignment_score"]
    stability = readiness_vector["stability_score"]
    symmetry = readiness_vector["symmetry_score"]
    readiness = readiness_vector["biomechanical_readiness_index"]
    bias = readiness_vector["structural_bias"]
    limit = readiness_vector["primary_limit_factor"]

    # -----------------------------
    # SMART INTERPRETATION LOGIC
    # -----------------------------

    def interpret_score(score):
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "moderate"
        else:
            return "poor"

    alignment_level = interpret_score(alignment)
    stability_level = interpret_score(stability)
    symmetry_level = interpret_score(symmetry)
    readiness_level = interpret_score(readiness)

    # -----------------------------
    # STRONG PROMPT (KEY UPGRADE)
    # -----------------------------

    prompt = f"""
You are an elite biomechanics expert and strength coach.

Analyze the athlete's movement data and generate a detailed, human-like coaching report keeping the Tone as described strictly.

------------------------
BIOMECHANICAL SCORES
------------------------
Alignment: {alignment:.2f} ({alignment_level})
Stability: {stability:.2f} ({stability_level})
Symmetry: {symmetry:.2f} ({symmetry_level})
Overall Readiness: {readiness:.2f} ({readiness_level})

Structural Bias: {bias}
Primary Limitation: {limit}

Recommended Exercises:
{exercises}

------------------------
INSTRUCTIONS
------------------------

1. Write a SMART and INSIGHTFUL summary (not generic)
   - Explain what the scores actually mean biomechanically
   - Mention movement quality, control, and injury risk
   -Keep it concise (2-3 sentences), but impactful

2. Identify 2–4 KEY ISSUES
   - Must be specific (e.g., "poor trunk alignment", not "low score")

3. Provide ACTIONABLE recommendations
   - Explain WHY each recommendation matters

4. Create a 3-week progression plan 
   - Week 1: correction phase
   - Week 2: control & stability
   - Week 3: strength integration

5. Tone:
   - Should be engaging and human-like, not robotic or generic
   - Easy to understand, even for non-experts
   - Like a real coach
   - Clear, confident, slightly motivational

------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------
{{
  "summary": "...",
  "key_issues": ["...", "..."],
  "recommendations": ["...", "..."],
  "progression_plan": {{
    "week_1": "...",
    "week_2": "...",
    "week_3": "..."
  }}
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8  
        )

        content = response.choices[0].message.content

        content = re.sub(r"```json|```", "", content).strip()

        parsed = json.loads(content)

        return parsed

    except Exception as e:
        return {
            "summary": "LLM generation failed",
            "key_issues": [],
            "recommendations": [],
            "progression_plan": {},
            "error": str(e)
        }