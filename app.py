import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

# models
url_1 = "https://drive.google.com/uc?id=1lRFipQVEaMIdoWU1MX8IpdSGqI_cPOZA"
model_1 = "model.h5"
gdown.download(url_1, output, quiet=False)

url_1 = "https://drive.google.com/uc?id=1NFRnfxCuT8-BVVgfTTkxhSYqgiQJXqMl"
model_1 = "model.h5"
gdown.download(url_1, output, quiet=False)

# loading models
@st.cache_resource
def load_models():
    model_csv = tf.keras.models.load_model(model_1)  
    model_image = tf.keras.models.load_model(model_2)
    return model_csv, model_image

model_csv, model_image = load_models()

# csv 
def predict_csv(input_df):
    if input_df.shape[1] != 8:
        st.error("CSV must have exactly 8 columns as per model input.")
        return None
    prediction = model_csv.predict(input_df)
    return prediction

# image spectrogram
def predict_image(img: Image.Image):
    img = img.resize((224, 224)) 
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2: 
        img_array = np.stack((img_array,)*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_image.predict(img_array)
    return (prediction[0][0] > 0.5)

# user interface 
st.title("Machine Vibration Anomaly Detection")
st.write("Upload either CSV raw vibration data or an image spectrogram to classify as normal or anomalous.")

option = st.radio("Choose input type", ['CSV Data', 'Spectrogram Image'])

if option == 'CSV Data':
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv, header=None)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
        if st.button("Predict"):
            prediction = predict_csv(df)
            if prediction is not None:
                result = 'Anomalous' if prediction[0] == 1 else 'Normal'
                st.success(f"Prediction: {result}")

elif option == 'Spectrogram Image':
    uploaded_img = st.file_uploader("Upload Spectrogram Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        st.image(image, caption='Uploaded Spectrogram', use_column_width=True)
        if st.button("Predict"):
            is_anomalous = predict_image(image)
            result = 'Anomalous' if is_anomalous else 'Normal'
            st.success(f"Prediction: {result}")
