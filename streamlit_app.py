import streamlit as st
from PIL import Image

st.title("Kamera Real-Time di Streamlit")

# Input dari kamera
camera_input = st.camera_input("Ambil Foto")

if camera_input:
    # Baca gambar dari input kamera
    img = Image.open(camera_input)
    st.image(img, caption="Gambar yang Diambil", use_column_width=True)
