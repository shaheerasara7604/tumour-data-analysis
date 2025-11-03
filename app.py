import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")
    return model

model = load_trained_model()

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image and let the model detect if a tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess the image
    img = img.resize((224, 224))  # 👈 change to your model’s input size if different
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if your model was trained with normalized data

    # Prediction
    prediction = model.predict(img_array)

    # Interpret result (adjust depending on your model output)
    if prediction[0][0] > 0.5:
        st.error("🚨 Tumor Detected!")
    else:
        st.success("✅ No Tumor Detected.")
