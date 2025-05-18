import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
from scipy.signal import stft
import joblib

# models
url_1 = "https://drive.google.com/uc?id=1lRFipQVEaMIdoWU1MX8IpdSGqI_cPOZA"
model_1 = "autoencoder.h5"
gdown.download(url_1, model_1, quiet=False)

url_2 = "https://drive.google.com/uc?id=1NFRnfxCuT8-BVVgfTTkxhSYqgiQJXqMl"
model_2 = "custom_dcnn_model.h5"
gdown.download(url_2, model_2, quiet=False)

# loading models
@st.cache_resource
def load_models():
    model_csv = tf.keras.models.load_model(model_1, compile=False)  
    model_image = tf.keras.models.load_model(model_2, compile=False)
    return model_csv, model_image

model_csv, model_image = load_models()

# preprocess csv
scaler = joblib.load("scaler.pkl")

def preprocess_uploaded_csv(df, downsample_factor=5):
    time_series = df.values[::downsample_factor]  # downsampled
    _, _, Zxx = stft(time_series.T, nperseg=64)
    freq_features = np.abs(Zxx).mean(axis=2).flatten()
    combined = np.hstack([time_series.flatten(), freq_features])
    combined_scaled = scaler.transform([combined])  # shape: (1, 400264)
    return combined_scaled

# csv 
def predict_csv(input_df):
    if input_df.shape[1] != 8:
        st.error("CSV must have exactly 8 columns (raw vibration channels).")
        return None
    try:
        input_data = preprocess_uploaded_csv(input_df)
        reconstruction = model_csv.predict(input_data)
        mse = np.mean(np.square(input_data - reconstruction), axis=1)
        threshold = 0.015  
        return (mse[0] > threshold)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

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
