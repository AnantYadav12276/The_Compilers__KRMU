import cv2
import numpy as np
import pandas as pd 
from datetime import datetime
import os
import requests

# Try to import face module from opencv-contrib-python
# If not available, the code will show an error message
try:
    # Check if face module is available (opencv-contrib-python)
    # Note: cv2.face is only available in opencv-contrib-python, not opencv-python
    if hasattr(cv2, 'face'):  # type: ignore[attr-defined]
        # OpenCV 4.x uses different function name
        # Try LBPHFaceRecognizer_create first (OpenCV 3.x-4.x legacy)
        if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):  # type: ignore[attr-defined]
            # Pylance may not recognize cv2.face, but this is valid at runtime
            # when opencv-contrib-python is installed
            LBPHFaceRecognizer_create = cv2.face.LBPHFaceRecognizer_create  # type: ignore
        elif hasattr(cv2.face, 'LBPHFaceRecognizer'):  # type: ignore[attr-defined]
            # OpenCV 4.x+ modern API
            LBPHFaceRecognizer_create = cv2.face.LBPHFaceRecognizer.create  # type: ignore
        else:
            print("WARNING: LBPHFaceRecognizer not found in cv2.face module")
            LBPHFaceRecognizer_create = None
    else:
        print("WARNING: cv2.face module not available. Install opencv-contrib-python")
        LBPHFaceRecognizer_create = None
except Exception as e:
    print(f"ERROR: Failed to initialize face recognizer: {e}")
    print("Please install opencv-contrib-python: pip install opencv-contrib-python")
    LBPHFaceRecognizer_create = None

SERVER_URL = "https://attendance-system-9ptl.onrender.com/upload-attendance"
API_KEY = "attendance_upload_key"
 
#* Load trained model
if LBPHFaceRecognizer_create is not None:
    recognizer = LBPHFaceRecognizer_create()
    # Check if model file exists before loading
    model_path = "Project/automated-attendance-system/facial_recognisition_model/trainer/trainer.yml"
    if os.path.exists(model_path):
        recognizer.read(model_path)
    else:
        print(f"WARNING: Model file not found at {model_path}")
        recognizer = None
else:
    recognizer = None

#* Load labels
label_map = {}
labels_path = "Project/automated-attendance-system/facial_recognisition_model/trainer/labels.txt"
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue  # Skip empty lines or malformed lines
            key, value = line.split(":", 1)  # Split only on first colon
            label_map[int(key)] = value
else:
    print(f"WARNING: Labels file not found at {labels_path}")

#* Face detector - use opencv's built-in haarcascade path
# Note: cv2.data.haarcascades exists at runtime but Pylance doesn't recognize it
haar_cascades_path = None
try:
    # This works at runtime but static type checkers may not recognize cv2.data
    haar_cascades_path = cv2.data.haarcascades  # type: ignore[attr-defined]
except AttributeError:
    pass

haar_paths = [
    os.path.join(haar_cascades_path, "haarcascade_frontalface_default.xml") if haar_cascades_path else "",
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",  # type: ignore[attr-defined]
    "C:/Users/Admin/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml",
    "C:/Program Files/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_default.xml"
]
face_cascade = None
for path in haar_paths:
    if os.path.exists(path):
        face_cascade = cv2.CascadeClassifier(path)
        break
if face_cascade is None:
    print("ERROR: Could not find haarcascade file. Please ensure opencv-python is installed correctly.")

# Camera settings - try different backends for cross-platform compatibility
cam = None
for index in [0, 1]:  # Try index 0 first (default webcam), then 1
    for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
        cam = cv2.VideoCapture(index, backend)
        if cam.isOpened():
            break
    if cam is not None and cam.isOpened():
        break
    if cam is not None:
        cam.release()
        cam = None

if cam is None or not cam.isOpened():
    print("ERROR: Could not open any camera. Please check camera index.")
    exit(1)

cam.set(3, 640)   # width
cam.set(4, 480)   # height



attendance_file = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"
# Create directory if it doesn't exist
attendance_dir = os.path.dirname(attendance_file)
if attendance_dir and not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir, exist_ok=True)
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time,Status\n")

marked_today = set()

# Check if recognizer and face_cascade are properly initialized
if recognizer is None or face_cascade is None:
    print("ERROR: Face recognition not initialized properly. Exiting.")
    exit(1)

while True:
    ret, frame = cam.read()
    if not ret:
        print("WARNING: Failed to capture frame. Retrying...")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray is None:
        continue
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        confidence_percentage = max(0, min(100, 100 - confidence))
        if confidence < 70:
            name = label_map.get(id_, "Unknown")
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        display_text = f"{name} ({confidence_percentage:.1f}%)"
        cv2.putText(
            frame,
            display_text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,255),
            2
        )

        if name != "Unknown":
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            key = (name, date)
            if key not in marked_today:
                marked_today.add(key)

                df = pd.read_csv(attendance_file)
                df.loc[len(df)] = [name, date, time, "Present"]
                df.to_csv(attendance_file, index=False)

                print(f"Attendance marked for {name}")

                #* Send attendance data to server

                payload = {
                    "Name": name,
                    "Date": date,
                    "Time": time,
                    "Status": "Present"
                }

                try:
                    requests.post(
                        SERVER_URL,
                        json=payload,
                        headers={"X-API-KEY": API_KEY},
                        timeout=5
                    )
                except Exception as e:
                    print("Upload failed:", e)


    cv2.imshow("Face Recognition Attendance", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

cam.release()
cv2.destroyAllWindows()


