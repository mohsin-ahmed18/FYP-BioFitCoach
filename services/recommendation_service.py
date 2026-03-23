from services.exercise_database import EXERCISE_DB


def rule_based_recommendations(readiness_vector):
    """
    Improved biomechanical rule engine with:
    - Priority-based selection (weakest area first)
    - Correct structural bias mapping
    - Proper movement balancing
    """

    alignment = readiness_vector["alignment_score"]
    stability = readiness_vector["stability_score"]
    symmetry = readiness_vector["symmetry_score"]
    structural_bias = readiness_vector["structural_bias"]
    limit = readiness_vector["primary_limit_factor"]

    candidate_pool = []

    # -----------------------------
    # PRIORITY: FIX WEAKEST AREA FIRST
    # -----------------------------
    if limit == "stability":
        candidate_pool += [
            "Goblet Squat",
            "Trap Bar Deadlift",
            "Farmer's Carry",
            "Cable Pallof Press",
            "Plank",
            "Single-Leg Romanian Deadlift"
        ]

    elif limit == "alignment":
        candidate_pool += [
            "Seated Cable Row",
            "Lat Pulldown",
            "Face Pull",
            "Barbell Hip Thrust",
            "Romanian Deadlift"
        ]

    elif limit == "symmetry":
        candidate_pool += [
            "Bulgarian Split Squat",
            "Walking Lunges",
            "Single-Arm Dumbbell Bench Press",
            "Single-Arm Cable Row",
            "Single-Leg Romanian Deadlift"
        ]

    # -----------------------------
    # SECONDARY SUPPORT LOGIC
    # -----------------------------
    if stability < 65:
        candidate_pool += [
            "Goblet Squat",
            "Farmer's Carry",
            "Plank"
        ]

    if alignment < 65:
        candidate_pool += [
            "Face Pull",
            "Romanian Deadlift"
        ]

    if symmetry < 65:
        candidate_pool += [
            "Walking Lunges",
            "Single-Leg Romanian Deadlift"
        ]

    # -----------------------------
    # STRUCTURAL BIAS (FIXED)
    # -----------------------------
    if structural_bias == "short_torso":
        candidate_pool += [
            "Box Squat",
            "Trap Bar Deadlift"
        ]

    elif structural_bias == "long_torso":
        candidate_pool += [
            "Barbell Front Squat",
            "Leg Press"
        ]

    elif structural_bias == "neutral":
        candidate_pool += [
            "Barbell Back Squat",
            "Conventional Deadlift",
            "Barbell Bench Press",
            "Barbell Row"
        ]

    # Remove duplicates
    candidate_pool = list(set(candidate_pool))

    # --------------------------------------
    # AUTO-BALANCE MOVEMENT PATTERNS (FIXED)
    # --------------------------------------

    balanced_selection = []
    movement_tracker = {
        "squat": [],
        "hinge": [],
        "push": [],
        "pull": [],
        "core": []
    }

    # ✅ FIXED LOOP
    for exercise in candidate_pool:
        movement_type = EXERCISE_DB.get(exercise)

        if movement_type and movement_type in movement_tracker:
            movement_tracker[movement_type].append(exercise)

    # Ensure at least one per category
    for movement, exercises in movement_tracker.items():
        if exercises:
            balanced_selection.append(exercises[0])

    # Fill remaining up to 8
    remaining = 8 - len(balanced_selection)

    for exercise in candidate_pool:
        if exercise not in balanced_selection and remaining > 0:
            balanced_selection.append(exercise)
            remaining -= 1

    return balanced_selection