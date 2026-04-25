class RepCounter:
    def __init__(self, down_threshold, up_threshold):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold

        self.stage = "up"
        self.count = 0

        # Tracking per rep
        self.min_angle = float("inf")
        self.max_angle = 0

    def update(self, angle):

        # Track range
        self.min_angle = min(self.min_angle, angle)
        self.max_angle = max(self.max_angle, angle)

        completed_rep = False

        if angle < self.down_threshold:
            self.stage = "down"

        if angle > self.up_threshold and self.stage == "down":
            self.stage = "up"
            self.count += 1
            completed_rep = True

            # Capture rep data
            rep_data = {
                "min_angle": self.min_angle,
                "max_angle": self.max_angle
            }

            # Reset for next rep
            self.min_angle = float("inf")
            self.max_angle = 0

            return self.count, self.stage, completed_rep, rep_data

        return self.count, self.stage, completed_rep, None