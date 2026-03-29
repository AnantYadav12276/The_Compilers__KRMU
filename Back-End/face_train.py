import cv2
import numpy as np
import os
from PIL import Image

# Import face module dynamically to avoid static type checker issues
# cv2.face is only available in opencv-contrib-python, not opencv-python
try:
    # Use dynamic import to avoid Pylance false positive
    face_module = __import__('cv2', fromlist=['face']).face  # type: ignore
    # Try OpenCV 4.x+ modern API first (preferred)
    if hasattr(face_module, 'LBPHFaceRecognizer'):
        LBPHFaceRecognizer_create = face_module.LBPHFaceRecognizer.create  # type: ignore
    # Fall back to OpenCV 3.x legacy API
    elif hasattr(face_module, 'LBPHFaceRecognizer_create'):
        LBPHFaceRecognizer_create = face_module.LBPHFaceRecognizer_create  # type: ignore
    else:
        print("ERROR: LBPHFaceRecognizer not found in cv2.face module")
        print("Please install opencv-contrib-python: pip install opencv-contrib-python")
        exit(1)
except (AttributeError, ImportError) as e:
    print("ERROR: cv2.face module not available")
    print("Please install opencv-contrib-python: pip install opencv-contrib-python")
    exit(1)

# Paths
dataset_path = "Project/automated-attendance-system/facial_recognisition_model/dataset"
trainer_path = "Project/automated-attendance-system/facial_recognisition_model/trainer"

# Create trainer folder if not exists
os.makedirs(trainer_path, exist_ok=True)

# Face detector - cv2.data.haarcascades exists at runtime but not recognized by static type checkers
# Use dynamic attribute access to suppress false positive warnings
data_module = __import__('cv2', fromlist=['data']).data  # type: ignore
haarcascades_path = getattr(data_module, 'haarcascades', None)
if haarcascades_path is None:
    raise RuntimeError("Could not find cv2.data.haarcascades. Please ensure OpenCV is properly installed.")

cascade_path = os.path.join(haarcascades_path, "haarcascade_frontalface_default.xml")

# Verify cascade file exists
if not os.path.isfile(cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found at: {cascade_path}")

detector = cv2.CascadeClassifier(cascade_path)

# Verify cascade loaded successfully
if detector.empty():
    raise ValueError(f"Failed to load Haar cascade file from: {cascade_path}")


def get_images_and_labels(path):
    face_samples = []
    ids = []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            try:
                img = Image.open(image_path).convert('L')
                img_np = np.array(img, 'uint8')

                faces = detector.detectMultiScale(img_np, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_samples.append(img_np[y:y+h, x:x+w])
                    ids.append(label_id)
            except Exception as e:
                print(f"Warning: Could not process image {image_path}: {e}")
                continue

        label_id += 1

    return face_samples, ids, label_map


print("Training faces...")

faces, ids, label_map = get_images_and_labels(dataset_path)

if len(faces) == 0:
    print("ERROR: No faces found. Run face_capture.py first.")
    exit()

recognizer = LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))

# Save model
model_path = os.path.join(trainer_path, "trainer.yml")
recognizer.save(model_path)

# Save labels
labels_path = os.path.join(trainer_path, "labels.txt")
with open(labels_path, "w") as f:
    for k, v in label_map.items():
        f.write(f"{k}:{v}\n")

print("Training completed successfully!")
print(f"Model saved to: {model_path}")
print(f"Labels saved to: {labels_path}")
