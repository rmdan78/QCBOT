import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

st.title("Camera Input and Image Upload with YOLOv5 Detection")

# Sidebar menu
option = st.sidebar.selectbox(
    'Select an option',
    ['Camera Input', 'Upload Image']
)

if option == 'Camera Input':
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

elif option == 'Upload Image':
    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        # Open and process image
        image = Image.open(uploaded_file)
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
        
