import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="TB Chest X-ray Classifier", layout="centered")
st.title("🫁 Tuberculosis Detection from Chest X-ray")
st.write("Upload an image to predict whether it's **Normal** or indicates **Tuberculosis (TB)**.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("resnet_model.h5")

model = load_model()

def preprocess(img):
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

uploaded_file = st.file_uploader("Upload X-ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_img = preprocess(image)
    prob = model.predict(processed_img)[0][0]

    st.info(f"Prediction Confidence: {prob:.2f} (Closer to 1 = TB, 0 = Normal)")


    if prob > 0.6:
        st.error("⚠️ The model predicts: **Tuberculosis (TB)**")
    else:
        st.success("✅ The model predicts: **Normal**")
