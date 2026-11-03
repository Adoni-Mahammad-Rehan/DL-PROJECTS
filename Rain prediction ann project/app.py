import json
import pickle
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Weather Prediction", page_icon="🌦️", layout="wide")

st.title("🌦️ Rain Tomorrow Prediction")
st.caption("Enter today's weather observations to estimate whether it will rain tomorrow.")

MODEL_PATH = "weather_model.keras"
SCALER_PATH = "scaler.pkl"
METADATA_PATH = "model_metadata.json"

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata

try:
    model, scaler, metadata = load_assets()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Keep weather_model.keras, scaler.pkl and model_metadata.json in the same folder as app.py.")
    st.stop()

encoded_columns = metadata["encoded_columns"]
numerical_columns = metadata["numerical_columns"]
categories = metadata["categories"]

# The model was trained with these weather measurements.
numeric_defaults = {
    "MinTemp": 12.0,
    "MaxTemp": 24.0,
    "Rainfall": 0.0,
    "Evaporation": 4.0,
    "Sunshine": 7.0,
    "WindGustSpeed": 35.0,
    "WindSpeed9am": 10.0,
    "WindSpeed3pm": 15.0,
    "Humidity9am": 70.0,
    "Humidity3pm": 50.0,
    "Pressure9am": 1015.0,
    "Pressure3pm": 1012.0,
    "Cloud9am": 4.0,
    "Cloud3pm": 4.0,
    "Temp9am": 17.0,
    "Temp3pm": 23.0,
}

st.subheader("Weather details")
with st.form("weather_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        selected_date = st.date_input("Date", value=date.today())
        location = st.selectbox("Location", categories["Location"])
        wind_gust_dir = st.selectbox("Wind gust direction", categories["WindGustDir"])
        wind_dir_9am = st.selectbox("Wind direction at 9am", categories["WindDir9am"])
        wind_dir_3pm = st.selectbox("Wind direction at 3pm", categories["WindDir3pm"])
        rain_today = st.selectbox("Rain today?", categories["RainToday"])

    with c2:
        st.markdown("**Temperature and rainfall**")
        min_temp = st.number_input("Minimum temperature (°C)", value=numeric_defaults["MinTemp"])
        max_temp = st.number_input("Maximum temperature (°C)", value=numeric_defaults["MaxTemp"])
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=numeric_defaults["Rainfall"])
        temp_9am = st.number_input("Temperature at 9am (°C)", value=numeric_defaults["Temp9am"])
        temp_3pm = st.number_input("Temperature at 3pm (°C)", value=numeric_defaults["Temp3pm"])
        evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=numeric_defaults["Evaporation"])
        sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=numeric_defaults["Sunshine"])

    with c3:
        st.markdown("**Wind, humidity and pressure**")
        wind_gust_speed = st.number_input("Wind gust speed (km/h)", min_value=0.0, value=numeric_defaults["WindGustSpeed"])
        wind_speed_9am = st.number_input("Wind speed at 9am (km/h)", min_value=0.0, value=numeric_defaults["WindSpeed9am"])
        wind_speed_3pm = st.number_input("Wind speed at 3pm (km/h)", min_value=0.0, value=numeric_defaults["WindSpeed3pm"])
        humidity_9am = st.number_input("Humidity at 9am (%)", min_value=0.0, max_value=100.0, value=numeric_defaults["Humidity9am"])
        humidity_3pm = st.number_input("Humidity at 3pm (%)", min_value=0.0, max_value=100.0, value=numeric_defaults["Humidity3pm"])
        pressure_9am = st.number_input("Pressure at 9am (hPa)", min_value=800.0, max_value=1100.0, value=numeric_defaults["Pressure9am"])
        pressure_3pm = st.number_input("Pressure at 3pm (hPa)", min_value=800.0, max_value=1100.0, value=numeric_defaults["Pressure3pm"])
        cloud_9am = st.number_input("Cloud at 9am (oktas)", min_value=0.0, max_value=9.0, value=numeric_defaults["Cloud9am"])
        cloud_3pm = st.number_input("Cloud at 3pm (oktas)", min_value=0.0, max_value=9.0, value=numeric_defaults["Cloud3pm"])

    submitted = st.form_submit_button("Predict rain tomorrow", type="primary", use_container_width=True)

if submitted:
    # Build the raw row using the same feature names used during training.
    row = {
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustSpeed": wind_gust_speed,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "Location": location,
        "WindGustDir": wind_gust_dir,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "RainToday": rain_today,
        "year": selected_date.year,
        "month": selected_date.month,
        "day": selected_date.day,
        "month_sin": np.sin(2 * np.pi * selected_date.month / 12),
        "month_cos": np.cos(2 * np.pi * selected_date.month / 12),
        "day_sin": np.sin(2 * np.pi * selected_date.day / 31),
        "day_cos": np.cos(2 * np.pi * selected_date.day / 31),
    }

    input_df = pd.DataFrame([row])

    categorical_columns = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
    input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

    # Match training-time one-hot column order and add any absent dummy columns as zero.
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Scale only the same 20 numerical columns, in the exact scaler training order.
    input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])
    input_encoded = input_encoded.astype("float32")

    expected_features = model.input_shape[-1]
    if input_encoded.shape[1] != expected_features:
        st.error(
            f"Feature mismatch: app created {input_encoded.shape[1]} features, "
            f"but model expects {expected_features}. Check the model and metadata files."
        )
        st.stop()

    probability = float(model.predict(input_encoded, verbose=0).ravel()[0])
    probability = float(np.clip(probability, 0.0, 1.0))

    st.divider()
    st.subheader("Prediction")
    left, right = st.columns(2)
    with left:
        if probability >= 0.5:
            st.error("🌧️ Rain is likely tomorrow")
        else:
            st.success("☀️ Rain is unlikely tomorrow")
    with right:
        st.metric("Predicted probability of rain", f"{probability:.1%}")
        st.progress(probability)

    st.caption("This is a machine-learning estimate, not an official weather forecast.")
