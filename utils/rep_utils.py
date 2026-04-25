class RepCounter:
    def __init__(self, down_threshold, up_threshold):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.stage = "up"
        self.count = 0

    def update(self, angle):
        if angle < self.down_threshold:
            self.stage = "down"

        if angle > self.up_threshold and self.stage == "down":
            self.stage = "up"
            self.count += 1

        return self.count, self.stage