import streamlit as st
from PIL import Image
import numpy as np
import torch

st.title("Camera Input with YOLOv5 Detection")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Camera input
camera = st.camera_input("Capture an image")

if camera:
    # Open and process image
    image = Image.open(camera)
    img_array = np.array(image)
    
    # Perform object detection
    results = model(img_array)

    # Draw bounding boxes on the image
    annotated_img = results.render()[0]  # Render bounding boxes on the image
    
    # Convert annotated image to PIL Image for Streamlit
    annotated_img_pil = Image.fromarray(annotated_img)
    
    # Display the annotated image
    st.image(annotated_img_pil, caption="Detected Objects", use_column_width=True)
