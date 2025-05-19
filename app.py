import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os
import joblib
from scipy.signal import stft

# URLs for models and scaler
url_1 = "https://drive.google.com/uc?id=1RjO6e_fI8NUT6F3zPs12qHYLOxqXIVOe"
model_1_path = "autoencoder.h5"

url_2 = "https://drive.google.com/uc?id=17jofX6sh8ennWJ_mboLyZxmcvJ2eniod"
model_2_path = "custom_dcnn_model.h5"

url_scaler = "https://drive.google.com/uc?id=1VS_8Se0KRqanfxYxTgLFi4fcv_lCEfjQ"
scaler_path = "scaler.pkl"

@st.cache_resource
def download_files():
    if not os.path.exists(model_1_path):
        gdown.download(url_1, model_1_path, quiet=False)
    if not os.path.exists(model_2_path):
        gdown.download(url_2, model_2_path, quiet=False)
    if not os.path.exists(scaler_path):
        gdown.download(url_scaler, scaler_path, quiet=False)

    scaler = joblib.load(scaler_path)
    return scaler

scaler = download_files()

# CSV preprocessing
def preprocess_csv(df, downsample_factor=5):
    if df.shape[1] != 8:
        st.error("CSV must have exactly 8 columns.")
        return None

    time_series = df.values[::downsample_factor]
    _, _, Zxx = stft(time_series.T, nperseg=64)
    freq_features = np.abs(Zxx).mean(axis=2).flatten()

    combined = np.hstack([time_series.flatten(), freq_features])
    combined = combined.reshape(1, -1)
    combined_scaled = scaler.transform(combined)
    return combined_scaled

@st.cache_resource
def load_models():
    model_csv = tf.keras.models.load_model(model_1_path, compile=False)
    model_image = tf.keras.models.load_model(model_2_path, compile=False)
    return model_csv, model_image

model_csv, model_image = load_models()

# CSV prediction
def predict_csv(input_df):
    processed = preprocess_csv(input_df)
    if processed is None:
        return None

    reconstruction = model_csv.predict(processed)
    mse = np.mean(np.square(processed - reconstruction), axis=1)
    threshold = 0.045960635265101094
    return (mse > threshold).astype(int)

# Image prediction
def predict_image(img: Image.Image):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_image.predict(img_array)
    return (prediction[0][0] > 0.5)

# Streamlit UI
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
            result = 'Faulty' if is_anomalous else 'Healthy'
            st.success(f"Prediction: {result}")
