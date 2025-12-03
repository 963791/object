import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(page_title="Object Detection & Tracking", layout="wide")

# -------------------------------
# Load high-accuracy YOLO model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8x.pt")   # Much more accurate than yolov8n or s

model = load_model()

# DeepSORT Tracker
tracker = DeepSort(max_age=10, n_init=2)


# -------------------------------
# VERY IMPORTANT — clean prediction method
# -------------------------------
def detect_objects(frame):
    # Preprocessing for accuracy
    frame = cv2.GaussianBlur(frame, (3,3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(
        frame,
        conf=0.45,          # higher confidence = fewer mistakes
        iou=0.5,            # improved NMS
        imgsz=640
    )

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())

            if conf < 0.45:
                continue

            detections.append([x1, y1, x2-x1, y2-y1, conf, cls])

    return detections, results


def track_objects(frame, detections):
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔍 Object Detection & Tracking – Ultra Accurate Version")

st.sidebar.header("Choose Input Source")
mode = st.sidebar.radio("Select Mode", ["Image Upload", "Video Upload", "Webcam Snapshot"])

# -----------------------------------
# IMAGE UPLOAD MODE
# -----------------------------------
if mode == "Image Upload":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        img = np.array(img)

        detections, results = detect_objects(img)
        tracks = track_objects(img, detections)

        # Draw results
        res = results[0].plot()

        st.image(res, caption="Detections", use_column_width=True)

# -----------------------------------
# VIDEO UPLOAD MODE
# -----------------------------------
elif mode == "Video Upload":
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_vid is not None:
        stframe = st.empty()

        video = cv2.VideoCapture(uploaded_vid.read())
        while True:
            ret, frame = video.read()
            if not ret:
                break

            detections, results = detect_objects(frame)
            tracks = track_objects(frame, detections)

            out = results[0].plot()
            stframe.image(out, channels="RGB")

# -----------------------------------
# WEBCAM SNAPSHOT MODE
# -----------------------------------
elif mode == "Webcam Snapshot":
    st.write("📸 Take a webcam picture for detection")

    img_data = st.camera_input("Take Picture")

    if img_data is not None:
        img = Image.open(img_data)
        img = np.array(img)

        detections, results = detect_objects(img)
        tracks = track_objects(img, detections)

        res = results[0].plot()
        st.image(res, caption="Detection Result", use_column_width=True)
