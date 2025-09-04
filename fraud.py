
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("creditcard_fraud_rf_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection")

st.markdown("""
Enter the **transaction details** below (V1 to V28 + Amount) to check whether it is **Fraudulent** or **Genuine**.
""")

# User inputs
input_data = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, step=0.01)
    input_data.append(val)

amount = st.number_input("Amount", min_value=0.0, step=0.01)
input_data.append(amount)

# Predict button
if st.button("🔍 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")

