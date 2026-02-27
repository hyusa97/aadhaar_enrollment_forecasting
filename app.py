import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Aadhaar Enrollment Predictor",
    page_icon="📊",
    layout="centered"
)

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #1c1f26;
            text-align: center;
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
rf = joblib.load("models/rf_model_new.pkl")
le = joblib.load("models/district_encoder.pkl")

# -------------------------
# Title Section
# -------------------------
st.title("📊 Aadhaar District Enrollment Prediction")
st.markdown("Short-term forecasting using Random Forest")

st.divider()

# -------------------------
# Input Section
# -------------------------
district_list = sorted(le.classes_)

col1, col2 = st.columns(2)

with col1:
    selected_district = st.selectbox("Select District", district_list)

with col2:
    month = st.selectbox("Select Month", list(range(1, 13)))

lag_value = st.number_input(
    "Previous Enrollment (lag_1)",
    min_value=0.0,
    step=1.0
)

st.divider()

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict Enrollment"):

    district_encoded = le.transform([selected_district])[0]

    input_data = pd.DataFrame({
        "lag_1": [lag_value],
        "month": [month],
        "district": [district_encoded]
    })

    prediction = rf.predict(input_data)[0]

    st.markdown(f"""
        <div class="prediction-box">
            Predicted Enrollment<br><br>
            <b>{round(prediction, 2)}</b>
        </div>
    """, unsafe_allow_html=True)