import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model for classification
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")  # Ensure 'best.pt' is a classification model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Streamlit UI
    st.title("Gallstone Classification with YOLOv8")
    st.write("Upload an ultrasound image to classify gallstones.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Load the model
            model = load_model()

            # Preprocess: Convert image to numpy array and resize to the model's input size
            image_array = np.array(image.resize((224, 224)))  # Use the size expected by the model

            # Predict using the YOLO classification model
            if model:
                results = model(image_array)
                
                # Check if results have probability information
                if len(results) > 0 and hasattr(results[0], 'probs'):
                    # Get the top prediction
                    top_class_index = results[0].probs.top1
                    predicted_class = results[0].names[top_class_index]
                    confidence = results[0].probs.top1conf.item()
                    
                    # Display results
                    st.write(f"**Prediction:** {predicted_class}")
                    st.write(f"**Confidence:** {confidence:.2f}")
                else:
                    st.error("Unable to process model predictions")
            else:
                st.error("Model is not loaded. Check the model path or logs.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
