import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

st.title("Camera Input with YOLO Detection")

# Load YOLO model
model = YOLO("yolov8n.pt")# Model YOLOv5 small

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

    # Display the annotated image
    st.image(annotated_img, caption="Detected Objects", use_column_width=True)
