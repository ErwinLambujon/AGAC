import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the models
@st.cache_resource
def load_ultrasound_model():
    try:
        return YOLO("ultrasound_model.pt")  # Path to the first model (Ultrasound vs Non-Ultrasound)
    except Exception as e:
        st.error(f"Error loading ultrasound model: {e}")
        return None

@st.cache_resource
def load_gallstone_model():
    try:
        return YOLO("secondmodel_best.pt")  # Path to the second model (Gallstone Classifier)
    except Exception as e:
        st.error(f"Error loading gallstone model: {e}")
        return None

def main():
    # Streamlit UI
    st.title("Gallstone Classification System")
    st.write("Upload an image to classify gallstones.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Load the models
            ultrasound_model = load_ultrasound_model()
            gallstone_model = load_gallstone_model()

            # Preprocess: Convert image to numpy array and resize to the model's input size (640x640)
            image_array = np.array(image.resize((640, 640)))  # Adjust size to fit the model input

            # Step 1: Classify if the image is ultrasound or non-ultrasound
            if ultrasound_model:
                ultrasound_results = ultrasound_model(image_array)
                
                # Check predictions for the first model
                if hasattr(ultrasound_results[0], 'probs'):
                    top_class_index = ultrasound_results[0].probs.top1
                    predicted_class = ultrasound_results[0].names[top_class_index]
                    confidence = ultrasound_results[0].probs.top1conf.item()

                    # Convert confidence to percentage
                    confidence_percentage = confidence * 100

                    # If it is an ultrasound image, classify for gallstones
                    if predicted_class.lower() == "ultrasound images" and confidence >= 0.9:  # Threshold confidence
                        
                        # Step 2: If it's an ultrasound image, classify for gallstones
                        if gallstone_model:
                            gallstone_results = gallstone_model(image_array)
                            
                            # Check predictions for gallstone model
                            if hasattr(gallstone_results[0], 'probs'):
                                gallstone_class_index = gallstone_results[0].probs.top1
                                gallstone_predicted_class = gallstone_results[0].names[gallstone_class_index]
                                gallstone_confidence = gallstone_results[0].probs.top1conf.item()

                                # Convert gallstone confidence to percentage
                                gallstone_confidence_percentage = gallstone_confidence * 100

                                # Display gallstone classification results
                                if gallstone_predicted_class == "cholelithiasis":
                                    st.write(f"**Gallstone Prediction:** Has gallstones")
                                else:
                                    st.write(f"**Gallstone Prediction:** No gallstones")
                                st.write(f"**Confidence:** {gallstone_confidence_percentage:.2f}%")
                    else:
                        image_confidence_percentage = confidence * 100
                        st.error(f"**Image Type:** {predicted_class} (Confidence: {image_confidence_percentage:.2f}%)")
                        st.error("Please upload a valid ultrasound image.")


                else:
                    st.error("Unable to process ultrasound model predictions.")
            else:
                st.error("Ultrasound model not loaded. Check the model path or logs.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
