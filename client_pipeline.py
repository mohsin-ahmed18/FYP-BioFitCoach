import requests
from capture.pose_capture import start_capture

BASE_URL = "http://127.0.0.1:8000"

# API CALLS

def create_session(name, email):
    payload = {
        "name": name,
        "email": email
    }

    res = requests.post(f"{BASE_URL}/session", json=payload)

    try:
        data = res.json()
    except:
        print("Invalid response:", res.text)
        return None

    print("DEBUG session_data:", data)

    if res.status_code != 200:
        print("Failed to create session:", data)
        return None

    return data


def upload_frames(session_id, frames):
    payload = {
        "session_id": session_id,
        "frames": frames
    }

    res = requests.post(f"{BASE_URL}/frames", json=payload)

    if res.status_code != 200:
        print("Frame upload failed:", res.text)
        return False

    return True


def run_analysis(session_id):
    payload = {"session_id": session_id}

    res = requests.post(f"{BASE_URL}/analysis", json=payload)

    try:
        return res.json()
    except:
        return {"error": res.text}


def generate_workout(session_id):
   
    res = requests.post(f"{BASE_URL}/generate-workout?session_id={session_id}")

    try:
        return res.json()
    except:
        return {"error": res.text}

# MAIN PIPELINE

def main():

    print("\nBioFitCoach System \n")

   
    name = input("Enter your name: ")
    email = input("Enter your email: ")


    session_data = create_session(name, email)

    if not session_data:
        return

    session_id = session_data.get("session_id")

    if not session_id:
        print(" session_id missing in response")
        return

    print(f"\nSession created successfully! ID: {session_id}")

    #  CAPTURE (WITH REPJUDGE)
    print("\nStarting pose capture...")
    print("Press 'q' to stop recording\n")

    frames = start_capture()

    if not frames or len(frames) == 0:
        print(" No frames captured")
        return

    print(f"Frames captured: {len(frames)}")

    #  UPLOAD FRAMES
    success = upload_frames(session_id, frames)

    if not success:
        return

    print("Frames uploaded successfully")

    # ANALYSIS
  
    print("\nBIOMECHANICAL ANALYSIS")
    print("")

    analysis = run_analysis(session_id)
    print(analysis)

    #  WORKOUT 

    print("\nSMART WORKOUT PLAN")
    print("")

    workout = generate_workout(session_id)
    print(workout)

    print("\nPipeline Executed Successfully!")

# RUN
if __name__ == "__main__":
    main()