# app_streamlit.py
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("linear_model_bodyfat.pkl")
scaler = joblib.load("scaler_bodyfat.pkl")

st.title("Body Fat Percentage Predictor")
st.write("Enter your body measurements below:")

# Input fields
age = st.number_input("Age", 18, 80, 30)
height_in = st.number_input("Height (inches)", 55, 80, 70)
neck = st.number_input("Neck circumference (cm)", 25, 55, 38)
chest = st.number_input("Chest circumference (cm)", 70, 140, 100)
abdomen = st.number_input("Abdomen circumference (cm)", 60, 150, 85)
hip = st.number_input("Hip circumference (cm)", 70, 150, 95)
thigh = st.number_input("Thigh circumference (cm)", 30, 90, 55)
knee = st.number_input("Knee circumference (cm)", 25, 55, 38)
ankle = st.number_input("Ankle circumference (cm)", 15, 35, 22)
biceps = st.number_input("Biceps circumference (cm)", 20, 50, 33)
forearm = st.number_input("Forearm circumference (cm)", 20, 40, 29)
wrist = st.number_input("Wrist circumference (cm)", 15, 25, 18)
weight_kg = st.number_input("Weight (kg)", 40, 150, 75)

if st.button("Predict Body Fat %"):
    # Prepare input
    input_data = np.array([[age, height_in, neck, chest, abdomen, hip, thigh,
                            knee, ankle, biceps, forearm, wrist, weight_kg]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Your estimated body fat percentage is: **{prediction:.2f}%**")