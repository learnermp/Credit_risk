import streamlit as st
import pickle
import os
import xgboost as xgb

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

MODEL_PATH = os.path.join("artifacts", "xgboost-model.bin")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            dv, model = pickle.load(f)
        return dv, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

dv, model = load_model()

if dv is None or model is None:
    st.stop()

# ==============================
# UI HEADER
# ==============================
st.title("Credit Risk Prediction")

st.markdown("""
This app predicts the probability of **loan default** using an XGBoost model.

Enter applicant details in the sidebar and click **Predict**.
""")

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("Applicant Information")

# Numeric Inputs
seniority = st.sidebar.number_input("Seniority (years)", 0, 50, 5)
time = st.sidebar.number_input("Loan Duration (months)", 6, 72, 48)
age = st.sidebar.number_input("Age", 18, 70, 30)

expenses = st.sidebar.number_input("Monthly Expenses", 0, 10000, 80)
income = st.sidebar.number_input("Monthly Income", 0, 100000, 150)
assets = st.sidebar.number_input("Assets", 0, 1000000, 5000)
debt = st.sidebar.number_input("Debt", 0, 1000000, 0)

amount = st.sidebar.number_input("Loan Amount", 100, 1000000, 1200)
price = st.sidebar.number_input("Purchase Price", 100, 1000000, 1600)

# Categorical Inputs (IMPORTANT: match training categories)
home = st.sidebar.selectbox("Home Ownership", ["owner", "rent", "parents"])
marital = st.sidebar.selectbox("Marital Status", ["married", "single", "widow", "divorced"])
records = st.sidebar.selectbox("Previous Default Records", ["no", "yes"])
job = st.sidebar.selectbox("Job Type", ["fixed", "freelance", "parttime"])

# ==============================
# BUILD INPUT
# ==============================
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

# ==============================
# PREDICTION
# ==============================
threshold = 0.3

if st.button("🔍 Predict"):

    # Basic validation
    if income == 0:
        st.warning("Income is zero. Prediction may be unreliable.")

    # Transform
    X_input = dv.transform([input_dict])
    dmatrix = xgb.DMatrix(X_input, feature_names=dv.feature_names_)

    # Predict
    prob = model.predict(dmatrix)[0]

    # ==============================
    # OUTPUT
    # ==============================
    st.subheader("Prediction Result")

    st.metric("Default Probability", f"{prob:.2%}")

    if prob >= threshold:
        st.error("High Risk: Loan should NOT be approved")
    else:
        st.success("Low Risk: Loan can be approved")

# ==============================
# FOOTER
# ==============================
st.markdown("---")

st.markdown("""
**Model Info**
- Algorithm: XGBoost  
- Threshold: 0.3  
- Task: Binary Classification (Default vs No Default)

**Note:** This tool is for educational/demo purposes.
""")