import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------
# 1. Load Model
# -----------------------------
@st.cache_resource
def load_model_func():
    return load_model("model.h5")
    

model = load_model_func()

# -----------------------------
# 2. Class Names
class_names=['Cat','Dog']
# ---------------------s
# 3. Streamlit
# -----------------------------
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image, and the model will predict whether it is a Dog or a Cat.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.success(f"Prediction: 🐶 Dog (Confidence: {confidence*100:.2f}%)")
    else:
        st.success(f"Prediction: 🐱 Cat (Confidence: {(1-confidence)*100:.2f}%)")
