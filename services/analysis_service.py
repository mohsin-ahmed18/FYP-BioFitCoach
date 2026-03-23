import numpy as np
import pandas as pd


# ---------------------------------------------------
# POPULATION STATISTICS
# ---------------------------------------------------

def compute_population_stats(df):

    df = df.copy()  # avoid modifying original dataframe

    # -----------------------------
    # 1️⃣ Derived Features
    # -----------------------------

    df["hip_knee_diff"] = abs(df["hip_angle"] - df["knee_angle"])

    df["shoulder_hip_ratio"] = df["shoulder_width"] / df["hip_width"]
    df["shoulder_hip_ratio"] = df["shoulder_hip_ratio"].replace(
        [float("inf"), -float("inf")], 0
    ).fillna(0)

    stats = {}

    # -----------------------------
    # 2️⃣ BASIC FEATURE STATS
    # -----------------------------

    base_features = [
        "body_angle",
        "hip_angle",
        "knee_angle",
        "elbow_angle",
        "torso_leg_ratio",
        "hip_knee_diff",
        "shoulder_hip_ratio"
    ]

    for feature in base_features:

        if feature not in df.columns:
            continue

        values = df[feature].dropna()

        if len(values) == 0:
            continue

        mean_val = values.mean()
        std_val = values.std()

        # ✅ prevent zero / nan std
        if pd.isna(std_val) or std_val < 1e-6:
            std_val = 1.0

        stats[feature] = {
            "mean": float(mean_val),
            "std": float(std_val)
        }

    # -----------------------------
    # 3️⃣ TRUE STABILITY STATS (IMPORTANT)
    # -----------------------------

    if "session_id" in df.columns:

        grouped = df.groupby("session_id")

        std_df = grouped.agg({
            "hip_angle": "std",
            "knee_angle": "std",
            "body_angle": "std"
        }).reset_index()

        std_df.rename(columns={
            "hip_angle": "std_hip_angle",
            "knee_angle": "std_knee_angle",
            "body_angle": "std_body_angle"
        }, inplace=True)

        # ✅ remove bad rows (very important)
        std_df = std_df.replace([float("inf"), -float("inf")], None)
        std_df = std_df.dropna()

        for feature in ["std_hip_angle", "std_knee_angle", "std_body_angle"]:

            if feature not in std_df.columns:
                continue

            values = std_df[feature].dropna()

            if len(values) == 0:
                continue

            mean_val = values.mean()
            std_val = values.std()

            # ✅ prevent collapse
            if pd.isna(std_val) or std_val < 1e-6:
                std_val = 1.0

            stats[feature] = {
                "mean": float(mean_val),
                "std": float(std_val)
            }

    # -----------------------------
    # 4️⃣ FALLBACK (if no session_id)
    # -----------------------------

    else:
        stats["std_hip_angle"] = {"mean": 5.0, "std": 2.0}
        stats["std_knee_angle"] = {"mean": 5.0, "std": 2.0}
        stats["std_body_angle"] = {"mean": 3.0, "std": 1.5}

    return stats


# ---------------------------------------------------
# Z SCORE
# ---------------------------------------------------

def z_score(value, mean, std):

    if std == 0:
        return 0

    z = (value - mean) / std

    # clamp
    return max(-3, min(z, 3))


# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

def compute_features(df):

    df["hip_knee_diff"] = abs(df["hip_angle"] - df["knee_angle"])

    df["shoulder_hip_ratio"] = df["shoulder_width"] / df["hip_width"]

    df["std_hip_angle"] = df["hip_angle"].std()
    df["std_knee_angle"] = df["knee_angle"].std()
    df["std_body_angle"] = df["body_angle"].std()

    return df


# ---------------------------------------------------
# ALIGNMENT SCORE
# ---------------------------------------------------

def compute_alignment_score(features, stats):

    body_z = abs(z_score(
        features["body_angle"],
        stats["body_angle"]["mean"],
        stats["body_angle"]["std"]
    ))

    ratio_z = abs(z_score(
        features["shoulder_hip_ratio"],
        stats["shoulder_hip_ratio"]["mean"],
        stats["shoulder_hip_ratio"]["std"]
    ))

    score = 100 - (body_z * 15 + ratio_z * 10)

    return max(0, min(score, 100))


# ---------------------------------------------------
# STABILITY SCORE
# ---------------------------------------------------

def compute_stability_score(features, stats):

    # fallback if missing
    if "std_hip_angle" not in stats:
        return 50

    hip = features["std_hip_angle"]
    knee = features["std_knee_angle"]
    body = features["std_body_angle"]

    hip_z = abs(z_score(hip, stats["std_hip_angle"]["mean"], stats["std_hip_angle"]["std"]))
    knee_z = abs(z_score(knee, stats["std_knee_angle"]["mean"], stats["std_knee_angle"]["std"]))
    body_z = abs(z_score(body, stats["std_body_angle"]["mean"], stats["std_body_angle"]["std"]))

    # ✅ Smooth penalty
    penalty = (hip_z * 6) + (knee_z * 6) + (body_z * 5)

    score = 100 - penalty

    # ✅ CRITICAL: prevent collapse
    return float(max(30, min(score, 100))) 


# ---------------------------------------------------
# SYMMETRY SCORE
# ---------------------------------------------------

def compute_symmetry_score(features, stats):

    diff_z = abs(z_score(
        features["hip_knee_diff"],
        stats["hip_knee_diff"]["mean"],
        stats["hip_knee_diff"]["std"]
    ))

    score = 100 - diff_z * 20

    return max(0, min(score, 100))


# ---------------------------------------------------
# STRUCTURAL BIAS
# ---------------------------------------------------

def classify_structure(torso_leg_ratio):

    if torso_leg_ratio < 0.85:
        return "Leg Dominant"

    if torso_leg_ratio > 1.1:
        return "Torso Dominant"

    return "Balanced"

def compute_session_features(df):

    features = {}

    # -----------------------------
    # MEAN FEATURES
    # -----------------------------
    features["body_angle"] = df["body_angle"].mean()
    features["hip_angle"] = df["hip_angle"].mean()
    features["knee_angle"] = df["knee_angle"].mean()
    features["elbow_angle"] = df["elbow_angle"].mean()

    features["shoulder_width"] = df["shoulder_width"].mean()
    features["hip_width"] = df["hip_width"].mean()
    features["torso_leg_ratio"] = df["torso_leg_ratio"].mean()

    # -----------------------------
    # ✅ CORRECT STABILITY FEATURES
    # -----------------------------
    features["std_hip_angle"] = float(df["hip_angle"].std() or 0)
    features["std_knee_angle"] = float(df["knee_angle"].std() or 0)
    features["std_body_angle"] = float(df["body_angle"].std() or 0)

    # -----------------------------
    # DERIVED
    # -----------------------------
    features["hip_knee_diff"] = abs(
        features["hip_angle"] - features["knee_angle"]
    )

    # Safe ratio
    if features["hip_width"] != 0:
        features["shoulder_hip_ratio"] = (
            features["shoulder_width"] / features["hip_width"]
        )
    else:
        features["shoulder_hip_ratio"] = 1.0

    return features
# ---------------------------------------------------
# COMPLETE ANALYSIS PIPELINE
# ---------------------------------------------------

def run_biomechanical_analysis(session_frames, population_frames):

    session_df = pd.DataFrame(session_frames)
    population_df = pd.DataFrame(population_frames)

    # ✅ Compute population stats (with derived features inside)
    population_stats = compute_population_stats(population_df)

    # ✅ Compute session features (single source of truth)
    features = compute_session_features(session_df)

    # ✅ Compute scores
    alignment = compute_alignment_score(features, population_stats)
    stability = compute_stability_score(features, population_stats)
    symmetry = compute_symmetry_score(features, population_stats)

    readiness = (alignment + stability + symmetry) / 3

    # ✅ Structural classification
    structural_bias = classify_structure(features["torso_leg_ratio"])

    # ✅ Limiting factor
    scores = {
        "alignment": alignment,
        "stability": stability,
        "symmetry": symmetry
    }

    limit_factor = min(scores, key=scores.get)

    return {
    "alignment_score": float(alignment),
    "stability_score": float(stability),
    "symmetry_score": float(symmetry),
    "biomechanical_readiness_index": float(readiness),
    "structural_bias": str(structural_bias),
    "primary_limit_factor": str(limit_factor)
}