import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model for classification
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ensure 'best.pt' is trained for classification

model = load_model()

# Streamlit UI
st.title("Gallstone Classification with YOLOv8")
st.write("Upload an ultrasound image to classify gallstones.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    image_array = np.array(image)

    # Predict using the YOLO classification model
    results = model.predict(image_array)

    # Extract the most probable class and its confidence
    predicted_class = results[0].names[np.argmax(results[0].probs)]
    confidence = np.max(results[0].probs)

    # Display results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
