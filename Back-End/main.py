import os
import sys

def run_command(command):
    result = os.system(command)
    if result != 0:
        print(f"❌ Error while running: {command}")
        sys.exit(1)

print("\n🚀 FACE RECOGNITION ATTENDANCE SYSTEM\n")

# Step 1: Capture Faces
capture_choice = input("Do you want to capture new faces? (y/n): ").lower()
trained_after_capture = False

if capture_choice == 'y':
    print("\n📸 Step 1: Capturing Faces...")
    run_command("python face_capture.py")

    # Automatically retrain after new faces
    print("\n🧠 Training updated model...")
    run_command("python face_train.py")
    trained_after_capture = True
else:
    print("⏭️ Skipping face capture...")

# Step 2: Optional manual training if no capture happened
if not trained_after_capture:
    train_choice = input("\nDo you want to train the model manually? (y/n): ").lower()
    if train_choice == 'y':
        print("\n🧠 Training Model...")
        run_command("python face_train.py")
    else:
        print("⏭️ Skipping training...")

# Step 3: Run Recognition
print("\n👁️ Step 3: Starting Attendance System...")
run_command("python recognition.py")

print("\n✅ System finished successfully!")