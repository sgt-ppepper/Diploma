import streamlit as st

# Page configuration
st.set_page_config(
    page_title="YOLO11s Video Analysis",
    page_icon="🎥",
    layout="wide"
)

#title = st.text_input("Diploma project interface", value="Diploma project interface")

# Display the title
#st.markdown(f"# {title}")
st.header("Diploma project interface")
# Placeholder for the image
st.image("AI.png", use_container_width=True)

# Instructions
st.write("Use the sidebar to navigate to different analysis pages.")