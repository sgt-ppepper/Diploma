import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
from pathlib import Path
import tempfile
import os

# Page configuration
st.set_page_config(page_title="EigenCAM Frame Analysis", page_icon="🔍")

# Header
st.header("Frame Analysis with EigenCAM")

# Sidebar for model and video configuration
st.sidebar.header("Configuration")
confidence_value = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# Load YOLO11s model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        #model.to("mps")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO11s model: {e}")
        return None

model = load_model()

# Video upload
st.sidebar.header("Video Upload")
uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    # Get total number of frames
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Error opening video file")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Frame number input
        st.sidebar.header("Frame Selection")
        frame_number = st.sidebar.number_input(
            "Enter Frame Number",
            min_value=1,
            max_value=total_frames,
            value=1,
            step=1
        )

        # Process button
        if st.sidebar.button("Process"):
            if model is None:
                st.error("Model not loaded. Please check model configuration.")
            else:
                # Open video and extract specified frame
                cap = cv2.VideoCapture(tfile.name)
                if not cap.isOpened():
                    st.error("Error opening video file")
                else:
                    # Set frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    ret, frame = cap.read()
                    if not ret:
                        st.error(f"Error reading frame {frame_number}")
                    else:
                        # Convert BGR to RGB
                        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Display original frame
                        st.subheader("Original Frame")
                        st.image(rgb_img, channels="RGB", use_container_width=True, caption=f"Frame {frame_number}")

                        # Process with EigenCAM
                        st.subheader("EigenCAM Visualization")
                        try:
                            # Prepare image for EigenCAM
                            img = np.float32(rgb_img) / 255
                            target_layers = [model.model.model[-3]]
                            cam = EigenCAM(model, target_layers, task='od')
                            grayscale_cam = cam(rgb_img)[0, :, :]
                            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
                            
                            # Display EigenCAM result
                            st.image(cam_image, channels="RGB", use_container_width=True, caption=f"EigenCAM for Frame {frame_number}")
                        except Exception as e:
                            st.error(f"Error processing EigenCAM: {e}")

                    # Release resources
                    cap.release()

                # Clean up
                os.unlink(tfile.name)
else:
    st.info("Please upload a video to begin processing.")