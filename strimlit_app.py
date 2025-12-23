import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Football Video YOLO", layout="centered")
st.title("⚽ Football Match Short Video Detection")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # ممكن تغيّري للموديل اللي عايزاه

model = load_model()

# Upload video (accept any common video format)
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("❌ Could not open the video file.")
    else:
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output temp video
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        # Input: عدد الثواني اللي عايزة تقصريها
        max_seconds = st.number_input("Length of short video (seconds)", min_value=1, max_value=30, value=10)
        max_frames = int(fps * max_seconds)

        stframe = st.empty()
        frame_count = 0
        st.info("Processing video... ⏳")

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection
            results = model(frame)
            frame = results[0].plot()

            # Add placeholder "Player" on each detected box
            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cv2.putText(frame, "Player", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            out.write(frame)
            frame_count += 1

            # Show every 5 frames for speed
            if frame_count % 5 == 0:
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        out.release()
        st.success("✅ Short video processing completed!")

        # Display final video
        st.video(out_file)
