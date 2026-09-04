import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

os.makedirs("models", exist_ok=True)

data = pd.read_csv(
    "data/student_data.csv"
)

X = data.drop(
    "final_score",
    axis=1
)

y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "models/baseline_model.pkl"
)

print("====================================")
print("BASELINE MODEL TRAINED")
print("====================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nModel:")
print("Random Forest Regressor")

print("\nNumber of trees:")
print(100)

print("\nModel saved at:")
print("models/baseline_model.pkl")