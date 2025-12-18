import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page config
st.set_page_config(page_title="YOLO Detection", layout="centered")

st.title("🔍 YOLO Object Detection Demo")
st.write("Upload an image and get YOLO detection results")

# Load model (cache to speed up)
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # خلي اسم الموديل بتاعك هنا

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Convert PIL image to OpenCV
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # YOLO inference
    results = model(img_array)

    # Plot results
    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.image(result_img, caption="Detection Result", use_container_width=True)
