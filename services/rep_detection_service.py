from utils.rep_utils import RepCounter


class RepJudge:
    def __init__(self):
        # Separate counters for exercises
        self.squat_counter = RepCounter(90, 160)
        self.pushup_counter = RepCounter(70, 150)

        self.current_exercise = "unknown"

    def detect_exercise(self, frame):
        knee = frame["knee_angle"]
        elbow = frame["elbow_angle"]
        body = frame["body_angle"]

        # ---- SIMPLE RULES ----
        if knee < 120 and body < 30:
            return "squat"

        if elbow < 100 and body > 30:
            return "push-up"

        return "unknown"

    def update(self, frame):
        exercise = self.detect_exercise(frame)
        reps = 0
        stage = ""

        if exercise == "squat":
            reps, stage = self.squat_counter.update(frame["knee_angle"])

        elif exercise == "push-up":
            reps, stage = self.pushup_counter.update(frame["elbow_angle"])

        self.current_exercise = exercise

        return {
            "exercise": exercise,
            "reps": reps,
            "stage": stage
        }