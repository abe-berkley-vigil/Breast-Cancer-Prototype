import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model and data ---
@st.cache_data
def load_model_and_data():
    model = joblib.load("models/model.pkl")
    
    # If you have training data saved, load it here. If not, mock example:
    try:
        X = pd.read_csv("X_train.csv")  # Replace with your actual feature dataset
    except FileNotFoundError:
        # Fallback dummy feature set for structure if actual data not available
        X = pd.DataFrame(np.random.rand(100, model.coef_.shape[1]),
                         columns=[f"feature_{i}" for i in range(model.coef_.shape[1])])
    
    return model, X

model, X = load_model_and_data()

st.title("Breast Cancer Diagnosis Predictor")
st.markdown("Adjust top 5 features to predict whether a tumor is **benign (0)** or **malignant (1)**.")

# --- Get feature importance ---
coeffs = model.coef_[0]
feature_importance = pd.Series(np.abs(coeffs), index=X.columns)
top_features = feature_importance.sort_values(ascending=False).head(5).index.tolist()
other_features = [col for col in X.columns if col not in top_features]

# --- User selection for fill method ---
fill_method = st.radio("Fill remaining features with:", options=["Mean", "Median"])

# --- Prepare input features ---
input_features = {}

st.subheader("Adjust Top 5 Influential Features")
for feature in top_features:
    input_features[feature] = st.slider(f"{feature}", 0.0, 1.0, 0.5, step=0.01)

# --- Fill remaining features ---
fill_values = X.median() if fill_method == "Median" else X.mean()
for feature in other_features:
    input_features[feature] = fill_values[feature]

# --- Predict ---
if st.button("Predict"):
    input_df = pd.DataFrame([input_features])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]
    
    label_map = {0: "benign", 1: "malignant"}
    
    st.success(f"**Prediction:** {prediction} - {label_map[prediction].capitalize()}")
    st.info(f"**Probability:** {probability:.4f}")
