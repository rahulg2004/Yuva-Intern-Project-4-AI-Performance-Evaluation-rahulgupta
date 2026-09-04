import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

os.makedirs("results", exist_ok=True)

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

model = joblib.load(
    "models/baseline_model.pkl"
)

predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)

print("====================================")
print("BASELINE MODEL EVALUATION")
print("====================================")

print("Mean Absolute Error (MAE):")
print(round(mae, 3))

print("\nRoot Mean Squared Error (RMSE):")
print(round(rmse, 3))

print("\nR2 Score:")
print(round(r2, 3))

print("\nNumber of test samples:")
print(len(y_test))

print("\nPerformance Interpretation:")

if mae < 5:
    print("MAE: Good prediction accuracy")
else:
    print("MAE: Prediction error can be improved")

if rmse < 7:
    print("RMSE: Good error performance")
else:
    print("RMSE: Large prediction errors may exist")

if r2 >= 0.80:
    print("R2: Strong model performance")
elif r2 >= 0.60:
    print("R2: Moderate model performance")
else:
    print("R2: Model requires improvement")

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel(
    "Actual Final Score"
)

plt.ylabel(
    "Predicted Final Score"
)

plt.title(
    "Baseline Model: Actual vs Predicted Scores"
)

plt.tight_layout()

plt.savefig(
    "results/baseline_prediction.png"
)

plt.show()

print("\nGraph saved at:")
print("results/baseline_prediction.png")