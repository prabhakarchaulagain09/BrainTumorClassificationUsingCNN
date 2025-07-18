
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Title and description
st.title("🧠 Brain Tumor Classification using CNN")
st.markdown("Upload a brain MRI image and this model will predict the type of tumor (if any).")
st.markdown("**Model Classes:** No Tumor, Pituitary Tumor, Glioma Tumor, Meningioma Tumor")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    return model

model = load_model()
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((150, 150))
    image = image.convert("RGB")
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 150, 150, 3)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.subheader("Prediction:")
    st.write(f"**{predicted_class}** with **{confidence:.2f}%** confidence.")

    # Show bar chart
    st.subheader("Confidence Scores:")
    score_dict = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    st.bar_chart(score_dict)
else:
    st.info("Please upload an MRI image to get started.")
