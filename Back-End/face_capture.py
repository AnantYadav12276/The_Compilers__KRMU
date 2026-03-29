import cv2 
import os

# Initialize camera - try index 0 first, then fallback to 1
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        raise RuntimeError("Failed to open camera. Please check camera connection and permissions.")

cam.set(3, 640)   # width
cam.set(4, 480)   # height

# Construct the path to the haarcascade file
# Using cv2.data.haarcascades with fallback for cross-platform compatibility
# Note: cv2.data.haarcascades exists in OpenCV but may not be recognized by static type checkers
try:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")  # type: ignore[attr-defined]
except (AttributeError, TypeError):
    # Fallback: construct path manually based on OpenCV installation directory
    cv2_path = cv2.__file__
    cv2_dir = os.path.dirname(cv2_path)
    data_dir = os.path.join(cv2_dir, "data", "haarcascades")
    cascade_path = os.path.join(data_dir, "haarcascade_frontalface_default.xml")

# Verify the cascade file exists before loading
if not os.path.isfile(cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found at: {cascade_path}")

face_detector = cv2.CascadeClassifier(cascade_path)

# Check if the cascade file was loaded successfully
if face_detector.empty():
    raise ValueError(f"Failed to load Haar cascade file from: {cascade_path}")

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

dataset_path = f"Project/automated-attendance-system/facial_recognisition_model/dataset/{student_name}"
os.makedirs(dataset_path, exist_ok=True)

count = 0
max_images = 40

while True:
    ret, img = cam.read()

    if not ret or img is None:
        print("❌ Failed to grab frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Only increment count once per frame if faces are detected
    if len(faces) > 0:
        count += len(faces)
        for (x, y, w, h) in faces:
            cv2.imwrite(
                f"{dataset_path}/{count}.jpg",
                gray[y:y+h, x:x+w]
            )
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Capture", img)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    elif count >= max_images:
        break   

cam.release()
cv2.destroyAllWindows()
