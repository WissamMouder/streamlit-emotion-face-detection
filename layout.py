# layout.py
import streamlit as st
import cv2
from PIL import Image
from pathlib import Path

def load_css():
    css_file = Path("assets/style.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_layout(predict_fn):
    st.set_page_config(page_title="Emotion Detection", layout="wide")

    # Load external CSS
    load_css()

    st.title("🎭 Emotion Detection")
    st.write("Detect emotions from facial expressions: Fear, Surprise, Angry, Sad, Happy")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=300)
            emotion, prob = predict_fn(image)
            st.markdown(
                f'<div class="prediction">Predicted Emotion: {emotion} ({prob*100:.2f}%)</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.header("Capture from Webcam")
        if st.button("Capture from Webcam"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                st.image(image, caption="Captured Image", width=300)
                emotion, prob = predict_fn(image)
                st.markdown(
                    f'<div class="prediction">Predicted Emotion: {emotion} ({prob*100:.2f}%)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.error("Failed to capture image from webcam")
