# 📊 Aadhaar District Enrollment Forecasting

An end-to-end machine learning pipeline designed to clean, analyze, and
forecast Aadhaar enrollment at the district level using historical
administrative data.

------------------------------------------------------------------------

## 📁 Project Structure

aadhaar_enrollment_forecasting/
│
├── app.py
├── models/
│   ├── rf_model.pkl
│   └── district_encoder.pkl
│
├── notebooks/
│   ├── data_review.ipynb
│   ├── data_clean.ipynb
│   ├── model_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore

------------------------------------------------------------------------

## 📌 Project Overview

The objective of this project is to provide a short-term forecasting
tool for Aadhaar enrollments. By leveraging administrative data, the
system helps identify enrollment trends and predicts future counts based
on historical persistence.

### Key Goals:

-   Clean and validate large-scale administrative data (500k+ rows)
-   Standardize inconsistent geographic identifiers (States/Districts)
-   Engineer time-based features (Lagging) for supervised learning
-   Deploy a memory-efficient model via Streamlit

------------------------------------------------------------------------

## 🔎 Data Cleaning & Problem Solving

### 1️⃣ State & District Standardization

**State Count Mismatch** - Initial count: 56\
- Expected count: 36

Issues: - Trailing spaces - Case inconsistencies (e.g., "bihar" vs
"Bihar") - Minor typos

Solution: - Standardized casing - Trimmed whitespace - Applied
dictionary mapping for corrections

------------------------------------------------------------------------

### 2️⃣ District & Pincode Structural Issue

Initial district count: 945\
Pincode count: 19,509

Observation: Multiple entries existed per district/date due to
pincode-level granularity.

Solution: Aggregated data using:

district + state + date → single row

Result: - One row per district per date - Clean chronological ordering -
Valid lag feature creation

------------------------------------------------------------------------

## 🧮 Feature Engineering

### Target Creation

total_enrollment = bio_age_5\_17 + bio_age_17\_

### Temporal Memory (Lag Feature)

lag_1 = previous enrollment value for the same district

This converted the time-series problem into a supervised learning task.

### Categorical Encoding

Districts were label encoded to allow the model to learn
district-specific behavioral differences.

------------------------------------------------------------------------

## 🤖 Modeling & Evaluation

### Attempt 1: Linear Regression

Features: - State - Year - Month

Result: R² = -27.9 (Model Failure)

Reason: - No temporal memory - Limited time depth (24 irregular dates) -
Inability to capture nonlinear district-level behavior

------------------------------------------------------------------------

### Final Model: Random Forest Regressor

Configuration: - n_estimators = 80 - max_depth = 20 - Time-based split
(80% train / 20% test) - No random shuffling to prevent leakage

### Performance Metrics

  Metric     Value
  ---------- -------
  MAE        \~175
  RMSE       \~313
  R² Score   0.72

### Feature Importance

-   lag_1 → \~86%
-   district → \~13%
-   month → negligible

Key Insight: Short-term enrollment persistence is the dominant
predictive signal.

------------------------------------------------------------------------

## 🚀 Deployment

The model is deployed via Streamlit.

Users can: 1. Select a district 2. Enter previous enrollment (lag_1) 3.
Select month 4. Receive instant prediction

### Deployment Optimization

Initial model exceeded GitHub's 50MB limit.

Solution: - Reduced number of trees - Limited tree depth - Applied
joblib compression

Final model size: \~43MB

------------------------------------------------------------------------

## 🏁 Key Learnings

-   Data structure matters more than raw row count
-   Lag features are critical in administrative forecasting
-   Linear models fail when temporal memory is absent
-   Tree-based models handle district-level heterogeneity effectively
-   Time-aware splitting prevents data leakage

------------------------------------------------------------------------

## 📌 Final Outcome

A production-ready district-level enrollment forecasting system
demonstrating:

-   Real-world data cleaning
-   Feature engineering for temporal ML
-   Model selection reasoning
-   Interactive Streamlit deployment
