import cv2
import numpy as np
import mediapipe as mp
import math
import time
from services.rep_detection_service import RepJudge

rep_judge = RepJudge()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# UTIL FUNCTIONS 

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)


def calc_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# PREPROCESSING 

def normalize_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image, gamma=1.25):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_frame(frame):
    frame = cv2.bilateralFilter(frame, 5, 50, 50)
    frame = normalize_lighting(frame)
    frame = apply_gamma_correction(frame)
    frame = cv2.resize(frame, (640, 480))
    return frame


# ONE EURO FILTER FOR SMOOTHING

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.007):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = None

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        x_hat = 0.9 * x + 0.1 * self.x_prev
        self.x_prev = x_hat
        return x_hat


# Filters
knee_filter = OneEuroFilter(60)
hip_filter = OneEuroFilter(60)
elbow_filter = OneEuroFilter(60)
torso_filter = OneEuroFilter(60)
shoulder_filter = OneEuroFilter(60)
hip_w_filter = OneEuroFilter(60)


# MAIN CAPTURE FUNCTION

def start_capture():

    cap = cv2.VideoCapture(0)
    frames_buffer = []

    with mp_pose.Pose() as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = preprocess_frame(frame)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:

               
                # DRAW SKELETON FIRST
                
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

           
                #  EXTRACT LANDMARKS
                
                lm = results.pose_landmarks.landmark

                # LEFT SIDE
                shoulder = [lm[11].x, lm[11].y, lm[11].z]
                elbow = [lm[13].x, lm[13].y, lm[13].z]
                wrist = [lm[15].x, lm[15].y, lm[15].z]
                hip = [lm[23].x, lm[23].y, lm[23].z]
                knee = [lm[25].x, lm[25].y, lm[25].z]
                ankle = [lm[27].x, lm[27].y, lm[27].z]

                # RIGHT SIDE
                shoulder_r = [lm[12].x, lm[12].y, lm[12].z]
                hip_r = [lm[24].x, lm[24].y, lm[24].z]

            
                #  CALCULATIONS
                
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)

                # BODY ANGLE
                dx = shoulder[0] - hip[0]
                dy = hip[1] - shoulder[1]
                body_angle = math.degrees(math.atan2(abs(dx), abs(dy)))

                # POSTURE CLASSIFICATION
                if body_angle is None:
                    posture_label = "Unknown"
                elif body_angle < 10:
                    posture_label = "Good"
                elif body_angle < 20:
                    posture_label = "Lean"
                else:
                    posture_label = "Slouch"

                # WIDTHS
                shoulder_width = calc_distance(shoulder, shoulder_r)
                hip_width = calc_distance(hip, hip_r)

                # RATIO
                leg_length = calc_distance(hip, ankle)
                torso_leg_ratio = (
                    calc_distance(shoulder, hip) / leg_length
                    if leg_length != 0 else 0
                )

                
                # SMOOTHED FRAME DATA
              
                frame_data = {
                    "elbow_angle": round(elbow_filter(elbow_angle), 2),
                    "hip_angle": round(hip_filter(hip_angle), 2),
                    "knee_angle": round(knee_filter(knee_angle), 2),
                    "shoulder_width": round(shoulder_filter(shoulder_width), 2),
                    "hip_width": round(hip_w_filter(hip_width), 2),
                    "torso_leg_ratio": round(torso_filter(torso_leg_ratio), 2),
                    "body_angle": round(body_angle, 2),
                    "posture_label": posture_label
                }

                frames_buffer.append(frame_data)
                
                rep_output = rep_judge.update(frame_data)
                
                #  OVERLAY UI
                
                cv2.rectangle(frame, (10, 50), (350, 230), (0, 0, 0), -1)

                cv2.putText(frame, f"Elbow: {frame_data['elbow_angle']}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                cv2.putText(frame, f"Hip: {frame_data['hip_angle']}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                cv2.putText(frame, f"Knee: {frame_data['knee_angle']}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                cv2.putText(frame, f"Body: {frame_data['body_angle']}", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                color = (0,255,0) if posture_label == "Good" else (0,165,255) if posture_label == "Lean" else (0,0,255)

                cv2.putText(frame, f"Posture: {posture_label}", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Exercise: {rep_output['exercise']}", (10, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                cv2.putText(frame, f"Reps: {rep_output['reps']}", (10, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(frame, f"Stage: {rep_output['stage']}", (10, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
          
            # FRAME COUNTER
            
            cv2.putText(frame, f"Frames: {len(frames_buffer)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

            cv2.imshow("BioFitCoach Capture", frame)

          
            # EXIT CONDITIONS
           
            MAX_FRAMES = 300

            if len(frames_buffer) >= MAX_FRAMES:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return frames_buffer