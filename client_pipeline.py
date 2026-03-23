import requests
from datetime import datetime
from capture.pose_capture import start_capture

BASE_URL = "http://127.0.0.1:8000"

# CREATE SESSION

def create_session(name, email):
    now = datetime.now()

    response = requests.post(
        f"{BASE_URL}/session",
        json={
            "name": name,
            "email": email,
            "session_date": now.strftime("%Y-%m-%d"),
            "capture_time": now.strftime("%H:%M:%S")
        }
    )

    return response.json()


# UPLOAD FRAMES

def upload_frames(session_id, frames):
    response = requests.post(
        f"{BASE_URL}/frames",
        json={
            "session_id": session_id,
            "frames": frames
        }
    )
    return response.json()


# RUN ANALYSIS

def run_analysis(session_id):
    response = requests.post(
        f"{BASE_URL}/analysis",
        json={"session_id": session_id}
    )
    return response.json()



#  GENERATE WORKOUT

def generate_workout(session_id):

    response = requests.post(
        f"{BASE_URL}/generate-workout",
        params={"session_id": session_id}   
    )

    return response.json()

# MAIN PIPELINE

def main():

    print("\n== BioFitCoach System ==\n")

    name = input("Enter your name: ").strip()
    email = input("Enter your email: ").strip()

    if not name or not email:
        print(" Name and Email are required!")
        return

   
    # CREATE SESSION
    
    session_data = create_session(name, email)
    print("DEBUG session_data:", session_data)

    session_id = session_data.get("session_id")

    if not session_id:
      print(" Failed to create session:", session_data)
      return
    print(f"\n Session created successfully! ID: {session_id}")

    
    #  CAPTURE FRAMES
    print("\n Starting pose capture...")
    print("Press 'q' to stop recording\n")

    frames = start_capture()

    if not frames:
        print(" No frames captured!")
        return

    print(f" Frames captured: {len(frames)}")

    #  UPLOAD FRAMES
    upload_frames(session_id, frames)
    print(" Frames uploaded successfully")

    
    #  RUN ANALYSIS
    analysis = run_analysis(session_id)

    print("\n BIOMECHANICAL ANALYSIS")
    print("---------------------------------")
    print(analysis)

   
    #  GENERATE WORKOUT
    workout = generate_workout(session_id)

    print("\nSMART WORKOUT PLAN")
    print("---------------------------------")
    print(workout)

    print("\n Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()