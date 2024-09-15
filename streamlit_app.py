import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = YOLO("yolo-Weights/yolov10n.pt")
    return model

model = load_model()

st.title("Camera Input with YOLOv5 Detection")

# Camera input
camera = st.camera_input("Capture an image")

if camera:
    # Open and process image
    image = Image.open(camera)
    img_array = np.array(image)

    # Convert image to RGB (YOLOv5 expects RGB images)
    img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(img_array_rgb)

    # Draw bounding boxes on the image
    annotated_img = results.render()[0]  # This will draw bounding boxes on the image

    # Convert numpy array to PIL Image for display
    annotated_img_pil = Image.fromarray(annotated_img)

    # Display the annotated image
    st.image(annotated_img_pil, caption="Detected Objects", use_column_width=True)

