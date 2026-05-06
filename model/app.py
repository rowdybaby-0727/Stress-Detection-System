import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stress Detection", layout="wide")

st.title("🧠 EEG / ECG Stress Detection Dashboard")

# Load models safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Machine learning models
main_model = joblib.load(os.path.join(BASE_DIR, "stress_model.pkl"))
eeg_model = joblib.load(os.path.join(BASE_DIR, "eeg_model.pkl"))

# Deep learning model
dl_model = load_model(os.path.join(BASE_DIR, "deep_stress_model.h5"))

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if uploaded:

    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Convert numeric safely
    df_num = df.apply(pd.to_numeric, errors='coerce')

    # Clean dataset
    df_num_clean = df_num.dropna(axis=1, how='all')
    df_num_clean = df_num_clean.dropna()

    # Decide model automatically
    if df_num_clean.shape[1] > 40:
        model = eeg_model
        st.success("EEG Model Selected")
    else:
        model = main_model
        st.success("ECG Model Selected")

    # Align features
    features = model.named_steps['scaler'].feature_names_in_
    aligned = pd.DataFrame(index=df_num_clean.index)

    for col in features:
        if col in df_num_clean.columns:
            aligned[col] = df_num_clean[col]
        else:
            aligned[col] = 0

    aligned = aligned.fillna(0)

    # Machine Learning Prediction
    pred_ml = model.predict(aligned)

    # Deep Learning Prediction
    pred_dl = dl_model.predict(aligned)
    pred_dl = (pred_dl > 0.5).astype(int).flatten()

    # Accuracy calculation (if label column exists)
    if "stress" in df.columns:
        y_true = df.loc[aligned.index, "stress"]
        acc_ml = accuracy_score(y_true, pred_ml)
        acc_dl = accuracy_score(y_true, pred_dl)

        st.subheader("Model Accuracy")
        st.write("Machine Learning Accuracy:", round(acc_ml,3))
        st.write("Deep Learning Accuracy:", round(acc_dl,3))

    # Add prediction results
    df["ML Prediction"] = "Not Predicted"
    df.loc[aligned.index, "ML Prediction"] = pd.Series(pred_ml).map(
        {0:"No Stress",1:"Stress"}
    ).values

    df["DL Prediction"] = "Not Predicted"
    df.loc[aligned.index, "DL Prediction"] = pd.Series(pred_dl).map(
        {0:"No Stress",1:"Stress"}
    ).values

    st.subheader("Prediction Results")
    st.dataframe(df)

    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "results.csv"
    )

else:
    st.info("Upload EEG or ECG dataset to begin")