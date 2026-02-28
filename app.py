import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Aadhaar Enrollment Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)
# ----------------------------
# Load Cleaned Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/upload/test_cleaned_final.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()
# ----------------------------
# Load Model + Encoder
# ----------------------------
rf = joblib.load("models/rf_model_new.pkl")
le = joblib.load("models/district_encoder.pkl")

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "📈 Analysis",
        "🧠 Forecasting",
        "🚨 Anomalies",
        "🗺 India Map"
    ]
)

# ----------------------------
# Section Routing
# ----------------------------

if section == "📊 Dashboard":
    st.title("📊 Overview Dashboard")

    # Total Enrollment
    total_enrollment = df['total_enrollment'].sum()
    st.metric("Total Enrollment (All Time)", f"{int(total_enrollment):,}")

    st.divider()

    # Enrollment by State
    state_totals = (
        df.groupby('state')['total_enrollment']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.subheader("Top 10 States by Enrollment")
    st.bar_chart(state_totals.head(10), x="state", y="total_enrollment")

    st.divider()

    # Time Trend
    daily_trend = (
        df.groupby('date')['total_enrollment']
        .sum()
        .reset_index()
    )

    st.subheader("Enrollment Trend Over Time")
    st.line_chart(daily_trend, x="date", y="total_enrollment")

elif section == "📈 State Analysis":
    st.title("📈 State-Level Analysis")
    st.write("State comparison charts will appear here.")

elif section == "🧠 Forecasting":
    st.title("🧠 District Forecasting")
    st.write("Forecasting interface will appear here.")

elif section == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.write("Anomalous enrollment patterns will appear here.")

elif section == "🗺 India Map":
    st.title("🗺 India Choropleth Map")
    st.write("Geographic visualization will appear here.")