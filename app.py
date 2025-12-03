import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

# -------------------------------
# Load YOLOv8 Model
# -------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8x-o365")
    return model

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("🟢 YOLOv8 Object Detection App")
st.sidebar.title("Settings")

mode = st.sidebar.radio("Choose Detection Mode:", ["Webcam", "Image", "Video"])

# -------------------------------
# Webcam Detection
# -------------------------------
if mode == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    if cap.isOpened():
        st.info("Press Ctrl+C in terminal to stop webcam.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read frame from webcam.")
                break
            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
    else:
        st.error("Webcam not detected.")

# -------------------------------
# Image Detection
# -------------------------------
elif mode == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        image_np = np.array(image)
        results = model.predict(image_np, conf=confidence)
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)

# -------------------------------
# Video Detection
# -------------------------------
elif mode == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        cap.release()
        out.release()
        st.success("Detection completed!")
        st.video(out_file.name)
