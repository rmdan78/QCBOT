import cv2
import streamlit as st

st.title("Menampilkan Video dari Kamera di Streamlit")

# Tempat untuk video
video_path = "output.mp4"

# Mengatur OpenCV untuk menangkap video
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Tidak dapat menangkap video dari kamera.")
        break

    # Menulis frame ke file video
    out.write(frame)

    # Menampilkan video
    st.video(video_path)

cap.release()
out.release()
