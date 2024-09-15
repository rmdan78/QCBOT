import streamlit as st
from PIL import Image
import numpy as np

st.title("Camera Input Test")

camera = st.camera_input("Capture an image")

if camera:
    image = Image.open(camera)
    img_array = np.array(image)

    st.image(img_array, caption="Captured Image", use_column_width=True)
