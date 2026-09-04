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

os.makedirs(
    "results",
    exist_ok=True
)

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

baseline_model = joblib.load(
    "models/baseline_model.pkl"
)

optimized_model = joblib.load(
    "models/optimized_model.pkl"
)

baseline_predictions = baseline_model.predict(
    X_test
)

optimized_predictions = optimized_model.predict(
    X_test
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

optimized_mae = mean_absolute_error(
    y_test,
    optimized_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

optimized_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        optimized_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

optimized_r2 = r2_score(
    y_test,
    optimized_predictions
)

result = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score"
    ],

    "Baseline": [
        baseline_mae,
        baseline_rmse,
        baseline_r2
    ],

    "Optimized": [
        optimized_mae,
        optimized_rmse,
        optimized_r2
    ]
})

result.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("====================================")
print("MODEL COMPARISON")
print("====================================")

print(
    result.to_string(
        index=False
    )
)

print("\n====================================")
print("IMPROVEMENT ANALYSIS")
print("====================================")

mae_improvement = (
    (baseline_mae - optimized_mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - optimized_rmse)
    / baseline_rmse
) * 100

r2_improvement = (
    (optimized_r2 - baseline_r2)
    / abs(baseline_r2)
) * 100

print(
    "MAE Improvement:",
    round(mae_improvement, 2),
    "%"
)

print(
    "RMSE Improvement:",
    round(rmse_improvement, 2),
    "%"
)

print(
    "R2 Improvement:",
    round(r2_improvement, 2),
    "%"
)

if optimized_mae < baseline_mae:
    print(
        "\nResult: Optimized model has lower MAE."
    )
else:
    print(
        "\nResult: MAE did not improve."
    )

if optimized_rmse < baseline_rmse:
    print(
        "Optimized model has lower RMSE."
    )
else:
    print(
        "RMSE did not improve."
    )

if optimized_r2 > baseline_r2:
    print(
        "Optimized model has higher R2 score."
    )
else:
    print(
        "R2 score did not improve."
    )

metrics = [
    "MAE",
    "RMSE",
    "R2 Score"
]

baseline_values = [
    baseline_mae,
    baseline_rmse,
    baseline_r2
]

optimized_values = [
    optimized_mae,
    optimized_rmse,
    optimized_r2
]

x = np.arange(
    len(metrics)
)

width = 0.35

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    optimized_values,
    width,
    label="Optimized"
)

plt.xlabel(
    "Performance Metrics"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Baseline vs Optimized Model Performance"
)

plt.xticks(
    x,
    metrics
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png"
)

plt.show()

print("\nComparison CSV saved at:")
print("results/model_comparison.csv")

print("\nComparison graph saved at:")
print("results/model_comparison.png")