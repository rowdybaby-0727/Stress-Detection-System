import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# Load dataset
data = pd.read_csv("D:\\stress project\\data\\balanced_40rows_dataset.csv")
# ECG columns
ecg_cols = [col for col in data.columns if "HR" in col or "AVNN" in col or "RMSSD" in col or "LF" in col or "HF" in col]

# EEG columns
eeg_cols = [col for col in data.columns if col not in ecg_cols + ["stress"]]

X = data[eeg_cols]
y = data["stress"]

# Model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Cross validation
scores = cross_val_score(model, X, y, cv=5)

print("EEG Cross Validation Accuracy:", scores.mean())

# Train final model
model.fit(X, y)

# Save model
joblib.dump(model, "eeg_model.pkl")