import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure best.pt is in the same directory

model = load_model()

# Streamlit UI
st.title("Gallstone Classification with YOLOv8")
st.write("Upload an ultrasound image to classify gallstones.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to array
    image_array = np.array(image)

    # Predict using the model
    results = model.predict(image_array)
    label = results[0].names[np.argmax(results[0].probs)]
    confidence = np.max(results[0].probs)

    # Display results
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")
