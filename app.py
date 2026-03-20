import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import xgboost as xgb

# Path to the saved model and vectorizer
MODEL_PATH = os.path.join("artifacts", "xgboost-model.bin")

# Load the model and DictVectorizer once when the app starts
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        dv, model = pickle.load(f)
    return dv, model

dv, model = load_model()

# Feature names (as used in the notebook)
feature_names = dv.feature_names_

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Risk Prediction")
st.markdown("""
This app predicts the probability of **loan default** based on applicant data.
The model was trained using XGBoost on a credit scoring dataset.
""")

# Sidebar for inputs
st.sidebar.header("Applicant Information")

# Numeric inputs
seniority = st.sidebar.number_input("Seniority (years)", min_value=0, max_value=50, value=5, step=1)
time = st.sidebar.number_input("Time (months)", min_value=6, max_value=72, value=48, step=6)
age = st.sidebar.number_input("Age", min_value=18, max_value=68, value=30, step=1)
expenses = st.sidebar.number_input("Monthly Expenses", min_value=0, value=80, step=10)
income = st.sidebar.number_input("Monthly Income", min_value=0, value=150, step=10)
assets = st.sidebar.number_input("Assets", min_value=0, value=5000, step=1000)
debt = st.sidebar.number_input("Debt", min_value=0, value=0, step=100)
amount = st.sidebar.number_input("Loan Amount", min_value=100, value=1200, step=100)
price = st.sidebar.number_input("Purchase Price", min_value=100, value=1600, step=100)

# Categorical inputs
home = st.sidebar.selectbox("Home Ownership", ["owner", "rent", "parents", "private", "ignore", "other", "unknown"])
marital = st.sidebar.selectbox("Marital Status", ["married", "single", "widow", "divorced", "separated", "unknown"])
records = st.sidebar.selectbox("Previous Records", ["no", "yes"])
job = st.sidebar.selectbox("Job Type", ["fixed", "freelance", "parttime", "others", "unknown"])

# Build a dictionary exactly as the DictVectorizer expects
input_dict = {
    "seniority": seniority,
    "home": home,
    "time": time,
    "age": age,
    "marital": marital,
    "records": records,
    "job": job,
    "expenses": expenses,
    "income": income,
    "assets": assets,
    "debt": debt,
    "amount": amount,
    "price": price,
}

# Transform the dictionary using the fitted DictVectorizer
X_input = dv.transform([input_dict])

# Create DMatrix for XGBoost prediction
dmatrix = xgb.DMatrix(X_input, feature_names=feature_names)

# Predict probability of default
prob = model.predict(dmatrix)[0]

# Classification based on threshold 0.3
threshold = 0.3
if prob >= threshold:
    pred = "There is chance of default: Loan should not be disbursed"
else:
    pred = "There is low chance of default: Loan can be disbursed"

# Display result
st.subheader("Prediction Result")
st.write(f"**Default probability:** {prob:.4f}")
st.write(pred)

st.markdown("---")
st.markdown("**Note:** A probability of default ≥ 0.3 is considered high risk; otherwise, the loan can be disbursed.")

# Data source information
with st.expander("Data Source"):
    st.markdown("""
    The model was trained on a **credit scoring dataset** used in the practical work of the  
    *Data Mining* course at the **Universitat Politècnica de Catalunya (UPC)**.  
    The original dataset was provided by the course instructors and is described on the course website:

    🔗 [UPC Data Mining Course – Materials for Practical Work](https://www.cs.upc.edu/%7Ebelanche/Docencia/mineria/mineria.html)
    """)