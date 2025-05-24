import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from pathlib import Path
import tempfile
import os

st.set_page_config(page_title="LIME Frame Analysis", page_icon="🧠")

st.header("Frame Analysis with LIME")

st.sidebar.header("Configuration")
confidence_value = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
num_samples = st.sidebar.slider("LIME Samples", 50, 500, 100, 10)

@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        model.to("mps")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO11s model: {e}")
        return None

model = load_model()

st.sidebar.header("Video Upload")
uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Error opening video file")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        st.sidebar.header("Frame Selection")
        frame_number = st.sidebar.number_input(
            "Enter Frame Number",
            min_value=1,
            max_value=total_frames,
            value=1,
            step=1
        )

        if model is None:
            st.error("Model not loaded. Please check model configuration.")
        else:
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Error opening video file")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, frame = cap.read()
                if not ret:
                    st.error(f"Error reading frame {frame_number}")
                else:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    results = model(image_rgb, conf=confidence_value)[0]
                    detections = results.boxes.data.cpu().numpy()

                    if len(detections) > 0:
                        detections = detections[detections[:, 4].argsort()[::-1]]
                        
                        object_options = [
                            f"{model.names[int(det[5])]} (Conf: {det[4]:.2f})"
                            for det in detections
                        ]
                        object_options = ["Select an object"] + object_options
                        
                        st.sidebar.header("Object Selection")
                        selected_object = st.sidebar.selectbox(
                            "Select Detected Object",
                            object_options
                        )

                        st.subheader("Original Frame")
                        st.image(image_rgb, channels="RGB", use_container_width=True, caption=f"Frame {frame_number}")

                        if selected_object != "Select an object" and st.sidebar.button("Process"):
                            selected_idx = object_options.index(selected_object) - 1
                            det = detections[selected_idx]
                            x1, y1, x2, y2, conf, cls = det.astype(int)
                            label = model.names[int(cls)]

                            crop_img = image_rgb[y1:y2, x1:x2]

                            st.subheader("Detected Object")
                            st.image(crop_img, channels="RGB", use_container_width=True, caption=f"Object: {label} (Conf: {conf:.2f})")

                            st.subheader("LIME Explanation")
                            try:
                                explainer = LimeImageExplainer()

                                def yolo_predict(images):
                                    preds = []
                                    for img in images:
                                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                        result = model(img_bgr, verbose=False)[0]
                                        boxes = result.boxes.data.cpu().numpy()
                                        cls_scores = np.zeros(len(model.names))
                                        for box in boxes:
                                            if int(box[5]) == int(cls) and box[4] > 0.5:
                                                cls_scores[int(cls)] = box[4]
                                        preds.append(cls_scores)
                                    return np.array(preds)

                                explanation = explainer.explain_instance(
                                    image_rgb,
                                    yolo_predict,
                                    top_labels=1,
                                    hide_color=0,
                                    num_samples=num_samples
                                )

                                temp, mask = explanation.get_image_and_mask(
                                    explanation.top_labels[0],
                                    positive_only=True,
                                    num_features=5,
                                    hide_rest=False
                                )

                                lime_image = mark_boundaries(image_rgb, mask)

                                st.image(lime_image, channels="RGB", use_container_width=True, caption=f"LIME Explanation for {label}")
                            except Exception as e:
                                st.error(f"Error processing LIME: {e}")

                    else:
                        st.warning("No objects detected in this frame.")
                
                cap.release()

            os.unlink(tfile.name)
else:
    st.info("Please upload a video to begin processing.")