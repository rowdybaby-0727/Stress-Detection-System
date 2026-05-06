import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("data/balanced_40rows_dataset.csv")
X = data.drop("stress", axis=1)
y = data["stress"]

# 80% training → model learns more
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Strong model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

joblib.dump(pipeline, "stress_model.pkl")