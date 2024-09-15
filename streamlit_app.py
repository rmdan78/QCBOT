import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

st.title("Camera Input with YOLOv8 Detection")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')  # Load YOLOv8 small model
    return model

model = load_model()

# Camera input
camera = st.camera_input("Capture an image")

if camera:
    # Open and process image
    image = Image.open(camera)
    img_array = np.array(image)

    # Perform object detection
    results = model(img_array)

    # Draw bounding boxes on the image
    annotated_img = results.plot()  # This will draw bounding boxes on the image

    # Convert numpy array to PIL Image for display
    annotated_img_pil = Image.fromarray(annotated_img)

    # Display the annotated image
    st.image(annotated_img_pil, caption="Detected Objects", use_column_width=True)

    # Display detection results
    st.write(results.pandas().xyxy[0])
