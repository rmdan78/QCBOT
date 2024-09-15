import streamlit as st
from PIL import Image, ImageDraw, ImageFont
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

    # Get detection results
    detections = results.pandas().xyxy[0]
    draw = ImageDraw.Draw(image)
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw bounding boxes
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class'])
        label = f"{model.names[cls]} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), label, fill="red", font=font)

    # Display the annotated image
    st.image(image, caption="Detected Objects", use_column_width=True)
