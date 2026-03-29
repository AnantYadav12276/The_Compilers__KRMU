# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import os

# # Note: cv2.face module requires opencv-contrib-python
# # Pylance may show warnings about 'face' not being a known attribute
# # but the code will work at runtime with proper OpenCV installation

# # Paths
# trainer_path = "Project/automated-attendance-system/facial_recognisition_model/trainer"
# attendance_file = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"

# # Create attendance folder
# os.makedirs(os.path.dirname(attendance_file), exist_ok=True)

# # Load model
# # Note: cv2.face is part of opencv-contrib-python, not in default cv2 type stubs
# # Using type: ignore to suppress Pylance warnings - code works at runtime
# try:
#     recognizer = cv2.face.LBPHFaceRecognizer.create()  # type: ignore
#     recognizer.read(f"{trainer_path}/trainer.yml")
# except AttributeError:
#     # Fallback for older OpenCV versions (3.x)
#     recognizer = cv2.face.LBPHFaceRecognizer_create()  # type: ignore
#     recognizer.read(f"{trainer_path}/trainer.yml")

# # Load labels
# label_map = {}
# with open(f"{trainer_path}/labels.txt", "r") as f:
#     for line in f:
#         key, value = line.strip().split(":")
#         label_map[int(key)] = value

# # Face detector - use cv2.data.haarcascades with fallback to absolute path
# # cv2.data exists at runtime but may not be in type stubs
# try:
#     cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")  # type: ignore[attr-defined]
# except AttributeError:
#     # Fallback to absolute path if cv2.data is not available
#     cascade_path = "C:/Users/Admin/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

# face_cascade = cv2.CascadeClassifier(cascade_path)

# # Camera
# cam = cv2.VideoCapture(0)

# if not cam.isOpened():
#     print("❌ Camera not working")
#     exit()

# # Create attendance file if not exists
# if not os.path.exists(attendance_file):
#     with open(attendance_file, "w") as f:
#         f.write("Name,Date,Time,Status\n")

# marked_today = set()

# print("👁️ Starting Face Recognition... Press 'q' to quit")

# while True:
#     ret, frame = cam.read()

#     if not ret:
#         print("❌ Failed to grab frame")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

#         if confidence < 80:
#             name = label_map.get(id_, "Unknown")
#         else:
#             name = "Unknown"

#         color = (0,255,0) if name != "Unknown" else (0,0,255)

#         cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
#         cv2.putText(frame, name, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

#         if name != "Unknown":
#             now = datetime.now()
#             date = now.strftime("%Y-%m-%d")
#             time = now.strftime("%H:%M:%S")

#             key = (name, date)

#             if key not in marked_today:
#                 marked_today.add(key)

#                 with open(attendance_file, "a") as f:
#                     f.write(f"{name},{date},{time},Present\n")

#                 print(f"✅ Attendance marked for {name}")

#     cv2.imshow("Face Recognition Attendance", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pyttsx3
# import mediapipe as mp

# ---------------- SETTINGS ----------------
trainer_path = "Project/automated-attendance-system/facial_recognisition_model/trainer"
attendance_file = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"

CONFIDENCE_THRESHOLD = 70

# Personalized Pam messages
custom_messages = {
    "Dhruv": "Hello Boss!",
    "Annie": "Hey Mommy!"
}

# ---------------- INIT ----------------
# Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(f"{trainer_path}/trainer.yml")

# Load labels
label_map = {}
with open(f"{trainer_path}/labels.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(":")
        label_map[int(key)] = value

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Attendance file
os.makedirs(os.path.dirname(attendance_file), exist_ok=True)
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time,Status\n")

marked_today = set()

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_draw = mp.solutions.drawing_utils

print("👁️ Smart Pam System Started... Press 'q' to quit")

# ---------------- LOOP ----------------
while True:
    ret, frame = cam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- FACE RECOGNITION --------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    current_names = []

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(id_, "Unknown")
        else:
            name = "Unknown"

        current_names.append(name)

        # Draw face box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Greeting text
        greeting = custom_messages.get(name, name)
        cv2.putText(frame, greeting, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # Attendance + voice (once per day)
        if name != "Unknown":
            today = datetime.now().strftime("%Y-%m-%d")
            key = (name, today)

            if key not in marked_today:
                marked_today.add(key)

                engine.say(greeting)
                engine.runAndWait()

                now = datetime.now()
                df = pd.read_csv(attendance_file)
                df.loc[len(df)] = [
                    name,
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    "Present"
                ]
                df.to_csv(attendance_file, index=False)

                print(f"✅ Attendance marked for {name}")

    # -------- HAND GESTURE (PAM TRIGGER) --------
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # result = hands.process(rgb_frame)

    # if result.multi_hand_landmarks:
    #     for hand_landmarks in result.multi_hand_landmarks:
    #         mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #         landmarks = hand_landmarks.landmark

    #         index_tip = landmarks[8]
    #         wrist = landmarks[0]

    #         # ✋ Hand raised condition
    #         if index_tip.y < wrist.y:
    #             cv2.putText(frame, "✋ Pam Activated!", (50,50),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    #             # Trigger Pam action
    #             if len(current_names) > 0 and current_names[0] != "Unknown":
    #                 person = current_names[0]
    #                 engine.say(f"Yes {person}, what do you want?")
    #             else:
    #                 engine.say("Yes, what do you want?")

    #             engine.runAndWait()

    # -------- DISPLAY --------
    cv2.imshow("Pam Face Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cam.release()
cv2.destroyAllWindows()