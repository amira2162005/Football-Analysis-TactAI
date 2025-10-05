import cv2
from roboflow import Roboflow
import os

# ============ STEP 1: Initialize Roboflow Football Action Model ============
rf = Roboflow(api_key="eqHlcwQJjQuMcBphKp3W")
project = rf.workspace("whdan").project("football-events-umfsc/1")
version = project.version(2)
action_model = version.model

# ============ STEP 2: Open Input Video ============
video_path = "input_videos/CV_Task.mkv"
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"❌ Could not open video: {video_path}")

frame_idx = 0
print("✅ Football Action Model loaded successfully. Starting frame analysis...")

# ============ STEP 3: Process Video Frames ============
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"🧩 Processing Frame {frame_idx}...")

    # Save frame temporarily to send to Roboflow model
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    try:
        # Run action detection on the full frame
        prediction = action_model.predict(temp_path).json()
        preds = prediction.get("predictions", [])

        # Draw detections on frame
        for p in preds:
            x, y, w, h = int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"])
            class_name = p["class"]
            conf = p["confidence"]

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    except Exception as e:
        print(f"⚠️ Error on frame {frame_idx}: {e}")

    # Save annotated frame (no display)
    output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"💾 Saved {output_path}")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ All processed frames saved in '{output_dir}/'")
