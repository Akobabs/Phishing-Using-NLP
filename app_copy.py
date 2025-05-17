import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")

st.title("E-commerce Fraud Prediction System")

st.markdown("Enter transaction details below:")

amount = st.number_input("Transaction Amount", min_value=0.0, max_value=10000.0)
hour = st.slider("Hour of Transaction", 0, 23)
transaction_type = st.selectbox("Transaction Type", ['purchase', 'refund', 'transfer'])
country = st.selectbox("Country", ['US', 'UK', 'NG', 'IN', 'CN'])

if st.button("Predict Fraud"):
    input_data = {
        'transaction_amount': amount,
        'time_hour': hour,
        'transaction_type_refund': 0,
        'transaction_type_transfer': 0,
        'country_IN': 0, 'country_NG': 0, 'country_UK': 0, 'country_US': 0
    }

    if transaction_type == 'refund':
        input_data['transaction_type_refund'] = 1
    elif transaction_type == 'transfer':
        input_data['transaction_type_transfer'] = 1

    if country in input_data:
        input_data[f'country_{country}'] = 1

    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.success(f"Prediction: {'FRAUD' if pred else 'LEGITIMATE'} | Probability: {prob:.2f}")
