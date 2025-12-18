import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Football Player Detection", layout="centered")
st.title("⚽ Football Match Player Detection")

# Load YOLO model (YOLOv8n small, جاهز)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Upload video
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Prepare output video file
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    stframe = st.empty()
    frame_count = 0
    st.info("Processing video, please wait... ⏳")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)
        frame = results[0].plot()  # plot boxes on frame

        # Add placeholder text "Player" on each detected box
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cv2.putText(frame, "Player", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        out.write(frame)
        frame_count += 1

        # Show current frame in Streamlit
        if frame_count % 5 == 0:  # update every 5 frames for speed
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()
    st.success("✅ Video processing completed!")

    # Display final video
    st.video(out_file)
