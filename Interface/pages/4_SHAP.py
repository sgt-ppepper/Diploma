import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import shap
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import io

st.set_page_config(page_title="SHAP Frame Analysis", page_icon="📊")

st.header("Frame Analysis with SHAP")

st.sidebar.header("Configuration")
confidence_value = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
shap_samples = st.sidebar.slider("SHAP Samples", 10, 1000, 50, 10)

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
                    
                    results = model(image_rgb, conf=confidence_value, verbose=False)[0]
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy()
                    class_names = results.names

                    if len(boxes) > 0:
                        top_indices = np.argsort(scores)[::-1]
                        detections = [
                            (boxes[i], scores[i], classes[i])
                            for i in top_indices
                        ]
                        
                        object_options = [
                            f"{class_names[int(cls)]} (Conf: {score:.2f})"
                            for _, score, cls in detections
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
                            box, score, cls = detections[selected_idx]
                            class_id = int(cls)
                            class_name = class_names[class_id]
                            box = box.astype(int)

                            crop_img = image_rgb[box[1]:box[3], box[0]:box[2]]

                            st.subheader("Detected Object")
                            st.image(crop_img, channels="RGB", use_container_width=True, caption=f"Object: {class_name} (Conf: {score:.2f})")

                            st.subheader("SHAP Explanation")
                            try:
                                def model_predict(images, target_box=box, target_class_id=class_id):
                                    predictions = []
                                    for img in images:
                                        img = img.transpose((1, 2, 0)).astype(np.uint8)
                                        result = model(img, verbose=False)
                                        detected_boxes = result[0].boxes.xyxy.cpu().numpy()
                                        conf = result[0].boxes.conf.cpu().numpy()
                                        cls = result[0].boxes.cls.cpu().numpy()
                                        best_iou = 0
                                        best_conf = 0
                                        for i, (det_box, det_conf, det_cls) in enumerate(zip(detected_boxes, conf, cls)):
                                            if int(det_cls) != target_class_id:
                                                continue
                                            x1 = max(target_box[0], det_box[0])
                                            y1 = max(target_box[1], det_box[1])
                                            x2 = min(target_box[2], det_box[2])
                                            y2 = min(target_box[3], det_box[3])
                                            intersection = max(0, x2 - x1) * max(0, y2 - y1)
                                            area1 = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                                            area2 = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                                            union = area1 + area2 - intersection
                                            iou = intersection / union if union > 0 else 0
                                            if iou > best_iou:
                                                best_iou = iou
                                                best_conf = det_conf
                                        predictions.append([best_conf])
                                    return np.array(predictions)

                                masker = shap.maskers.Image("inpaint_telea", image_rgb.shape)

                                explainer = shap.Explainer(model_predict, masker)

                                img_shap = image_rgb.transpose((2, 0, 1))[np.newaxis, ...]

                                shap_values = explainer(img_shap, max_evals=shap_samples)

                                shap_vals = shap_values.values[0, ..., 0]
                                shap_vals = np.transpose(shap_vals, (1, 2, 0))

                                plt.figure(figsize=(8, 6))
                                plot = shap.image_plot(shap_vals, pixel_values=image_rgb, labels=[class_name], show=False)
                                
                                buf = io.BytesIO()
                                plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                                plt.close()
                                buf.seek(0)
                                shap_image = Image.open(buf)

                                st.image(shap_image, use_container_width=True, caption=f"SHAP Explanation for {class_name}")
                            except Exception as e:
                                st.error(f"Error processing SHAP: {e}")

                    else:
                        st.warning("No objects detected in this frame.")
                
                cap.release()

            os.unlink(tfile.name)
else:
    st.info("Please upload a video to begin processing.")