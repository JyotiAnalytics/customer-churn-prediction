"""Streamlit churn predictor. Run: streamlit run app.py  (after python main.py)"""
import joblib
import pandas as pd
import streamlit as st

from main import add_features

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


model = load_model()

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
with c2:
    tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    phone = st.selectbox("Phone service", ["Yes", "No"])
    lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

if st.button("Predict churn risk"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": tech, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure if tenure > 0 else float("nan"),
    }])
    X = add_features(row)
    p = float(model.predict_proba(X)[0, 1])
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption("Model output from the trained pipeline; it indicates association, not cause.")
