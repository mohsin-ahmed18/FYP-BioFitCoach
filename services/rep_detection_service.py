import numpy as np
from utils.rep_utils import RepCounter


class RepJudge:
    def __init__(self):

        # REP COUNTERS
        self.squat_counter = RepCounter(90, 160)
        self.pushup_counter = RepCounter(70, 150)

        
        # STATE TRACKING
        self.current_exercise = "unknown"
        self.prev_angles = []

        # Rep scoring memory
        self.rep_scores = []
        self.last_feedback = ""
       
    
    # EXERCISE DETECTION
    def detect_exercise(self, frame):

        knee = frame["knee_angle"]
        elbow = frame["elbow_angle"]
        body = frame["body_angle"]

        if knee < 120 and body < 30:
            return "squat"

        if elbow < 100 and body > 30:
            return "push-up"

        return "unknown"

    
    # SMOOTHNESS (CONTROL)
    def compute_smoothness(self):

        if len(self.prev_angles) < 5:
            return 1.0  # assume good

        diffs = np.diff(self.prev_angles)
        jerk = np.std(diffs)

        # Lower jerk = better control
        score = max(0, min(1, 1 - (jerk / 20)))

        return score

    # SQUAT SCORING
    def score_squat(self, min_knee_angle, smoothness, body_angle):

        score = 100
        feedback = []

        # DEPTH
        if min_knee_angle > 100:
            score -= 30
            feedback.append("Go deeper")

        elif min_knee_angle > 90:
            score -= 10
            feedback.append("Slightly deeper")

        else:
            feedback.append("Good depth")

        
        # CONTROL
        if smoothness < 0.5:
            score -= 25
            feedback.append("Control movement")

        else:
            feedback.append("Good control")

        # ALIGNMENT 
        
        if body_angle > 25:
            score -= 15
            feedback.append("Keep chest upright")

        # Clamp
        score = max(0, min(score, 100))

        return score, feedback

    
    # PUSHUP SCORING

    def score_pushup(self, min_elbow_angle, smoothness, body_angle):

        score = 100
        feedback = []

        # Depth
        if min_elbow_angle > 100:
            score -= 30
            feedback.append("Go lower")

        elif min_elbow_angle > 80:
            score -= 10
            feedback.append("Slightly lower")

        else:
            feedback.append("Good depth")

        # Control
        if smoothness < 0.5:
            score -= 25
            feedback.append("Control movement")

        else:
            feedback.append("Good control")

        # Body alignment
        if body_angle > 40:
            score -= 15
            feedback.append("Keep body straight")

        score = max(0, min(score, 100))

        return score, feedback

   
    # MAIN UPDATE FUNCTION

    def update(self, frame):

        exercise = self.detect_exercise(frame)

        reps = 0
        stage = ""
        score = None
        feedback = []

        # STORE ANGLES FOR SMOOTHNESS
        if exercise == "squat":
            self.prev_angles.append(frame["knee_angle"])
        elif exercise == "push-up":
            self.prev_angles.append(frame["elbow_angle"])

        if len(self.prev_angles) > 20:
            self.prev_angles.pop(0)

        smoothness = self.compute_smoothness()

        # SQUAT LOGIC
        if exercise == "squat":

            reps, stage = self.squat_counter.update(frame["knee_angle"])

            # Detect rep completion
            if stage == "up" and len(self.prev_angles) > 5:

                min_knee = min(self.prev_angles)

                score, feedback = self.score_squat(
                    min_knee,
                    smoothness,
                    frame["body_angle"]
                )

                self.rep_scores.append(score)
                self.last_feedback = ", ".join(feedback)

                self.prev_angles.clear()

        
        # PUSHUP LOGIC
        elif exercise == "push-up":

            reps, stage = self.pushup_counter.update(frame["elbow_angle"])

            if stage == "up" and len(self.prev_angles) > 5:

                min_elbow = min(self.prev_angles)

                score, feedback = self.score_pushup(
                    min_elbow,
                    smoothness,
                    frame["body_angle"]
                )

                self.rep_scores.append(score)
                self.last_feedback = ", ".join(feedback)

                self.prev_angles.clear()

        #  OUTPUT
        avg_score = (
            sum(self.rep_scores) / len(self.rep_scores)
            if self.rep_scores else 0
        )

        return {
            "exercise": exercise,
            "reps": reps,
            "stage": stage,
            "rep_score": score,
            "avg_score": round(avg_score, 2),
            "feedback": self.last_feedback
        }