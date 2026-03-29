import threading
import cv2
import numpy as np
import pandas as pd 
from datetime import datetime
import os
import requests
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# Enable CORS for all routes
CORS(app)

# Global variable to store recognized name
recognized_name = "Unknown"
recognition_active = False

# Try to import face module from opencv-contrib-python
# Note: cv2.face is only available in opencv-contrib-python, not opencv-python
# The type ignore comments suppress Pylance warnings about cv2.face not being in type stubs
LBPHFaceRecognizer_create = None
try:
    # Import the face submodule - may fail if opencv-contrib-python not installed
    from cv2 import face  # type: ignore[attr-defined]
    
    # Try both creation methods (newer API uses .create(), older uses direct function)
    if hasattr(face, 'LBPHFaceRecognizer_create'):
        LBPHFaceRecognizer_create = face.LBPHFaceRecognizer_create
    elif hasattr(face, 'LBPHFaceRecognizer'):
        LBPHFaceRecognizer_create = face.LBPHFaceRecognizer.create
    else:
        print("WARNING: LBPHFaceRecognizer not found in cv2.face module")
except ImportError as e:
    print(f"WARNING: cv2.face module not available. Install opencv-contrib-python: {e}")
except AttributeError as e:
    print(f"WARNING: cv2.face module not accessible: {e}")
except Exception as e:
    print(f"ERROR: Failed to initialize face recognizer: {e}")

# Load trained model
recognizer = None
if LBPHFaceRecognizer_create is not None:
    recognizer = LBPHFaceRecognizer_create()
    model_path = "Project/automated-attendance-system/facial_recognisition_model/trainer/trainer.yml"
    if os.path.exists(model_path):
        recognizer.read(model_path)
    else:
        print(f"WARNING: Model file not found at {model_path}")

# Load labels
label_map = {}
labels_path = "Project/automated-attendance-system/facial_recognisition_model/trainer/labels.txt"
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(":", 1)
            label_map[int(key)] = value

# Face detector
haar_cascades_path = None
try:
    haar_cascades_path = cv2.data.haarcascades  # type: ignore[attr-defined]
except AttributeError:
    pass

haar_paths = [
    os.path.join(haar_cascades_path, "haarcascade_frontalface_default.xml") if haar_cascades_path else "",
    "C:/Users/Admin/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_default.xml"
]
face_cascade = None
for path in haar_paths:
    if os.path.exists(path):
        face_cascade = cv2.CascadeClassifier(path)
        break

attendance_file = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"
attendance_dir = os.path.dirname(attendance_file)
if attendance_dir and not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir, exist_ok=True)
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time,Status\n")


def run_face_recognition():
    """Run face recognition in a separate thread"""
    global recognized_name, recognition_active
    
    if recognizer is None or face_cascade is None:
        print("ERROR: Face recognition not initialized properly")
        recognition_active = False
        return
    
    # Try different camera backends
    cam = None
    for index in [0, 1]:
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
        print("ERROR: Could not open camera")
        recognition_active = False
        return
    
    cam.set(3, 640)
    cam.set(4, 480)
    
    marked_today = set()
    
    while recognition_active:
        ret, frame = cam.read()
        if not ret:
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
            
            recognized_name = name
            
            # Draw rectangle and label on frame
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            display_text = f"{name} ({confidence_percentage:.1f}%)"
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            
            # Mark attendance
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
        
        # Show the frame (optional - for debugging)
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/styles.css')
def serve_css():
    return send_from_directory('frontend', 'styles.css')


@app.route('/app.js')
def serve_js():
    return send_from_directory('frontend', 'app.js')


@app.route('/start-face-recognition')
def start_face():
    global recognized_name, recognition_active
    
    if recognition_active:
        return jsonify({"name": recognized_name})
    
    recognition_active = True
    recognized_name = "Scanning..."
    
    # Start face recognition in a separate thread
    try:
        thread = threading.Thread(target=run_face_recognition)
        thread.daemon = True
        thread.start()
        return jsonify({"name": "Recognition started"})
    except Exception as e:
        print(f"Error starting recognition: {e}")
        return jsonify({"name": f"Error: {str(e)}"}), 500


@app.route('/stop-face-recognition')
def stop_face_recognition():
    global recognition_active
    recognition_active = False
    return jsonify({"name": recognized_name})


@app.route('/get-recognized-name')
def get_recognized_name():
    return jsonify({"name": recognized_name})


if __name__ == '__main__':
    app.run(debug=True, port=5000)