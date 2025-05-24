import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os
import time

# Page configuration
st.set_page_config(page_title="YOLO11s Video Detection", page_icon="🎥")

# Header
st.header("Video Object Detection with YOLO11s")

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
confidence_value = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# Load YOLO11s model
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

# Video upload
st.sidebar.header("Video Upload")
uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    # Display original video
    #st.subheader("Original Video")
    #st.video(tfile.name)
    frame_counter = 0
    # Process video button
    if st.sidebar.button("Process Video"):
        if model is None:
            st.error("Model not loaded. Please check model configuration.")
        else:
            st.subheader("Processed Video")
            st_frame = st.empty()
            
            # Open video
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Error opening video file")
            else:
                # Get video properties
                #fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 
                # Create temporary file for processed video
                #output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                #output_file.close()
                
                # Define video writer
                #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                #out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
                prev_time = time.time()
                # Process video frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    #fps = cap.get(cv2.CAP_PROP_FPS)
                    curr_time = time.time()
                    frame_time = curr_time - prev_time
                    fps = int(1.0 / frame_time) if frame_time > 0 else 0
                    prev_time = curr_time
                    frame_counter += 1
                    # Run YOLO11s prediction
                    results = model.predict(frame, conf=confidence_value)
                    annotated_frame = results[0].plot()
                    
                    # Write to output video
                    #out.write(annotated_frame)
                    
                    # Display frame
                    st_frame.image(annotated_frame, channels="BGR", use_container_width=True, caption=f"Кадр {frame_counter}; FPS = {fps}")
                
                # Release resources
                cap.release()
                #out.release()
                
                # Display processed video
                # st.subheader("Download Processed Video")
                # with open(output_file.name, 'rb') as f:
                #     st.download_button(
                #         label="Download Processed Video",
                #         data=f,
                #         file_name="processed_video.mp4",
                #         mime="video/mp4"
                #     )
                
                # Clean up
                os.unlink(tfile.name)
                #os.unlink(output_file.name)
else:
    st.info("Please upload a video to begin processing.")