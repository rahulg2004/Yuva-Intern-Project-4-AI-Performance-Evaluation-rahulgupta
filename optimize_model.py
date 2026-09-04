import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

os.makedirs(
    "models",
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

model = RandomForestRegressor(
    random_state=42
)

parameters = {
    "n_estimators": [
        50,
        100,
        200
    ],

    "max_depth": [
        None,
        5,
        10,
        20
    ],

    "min_samples_split": [
        2,
        5,
        10
    ]
}

print("====================================")
print("STARTING MODEL OPTIMIZATION")
print("====================================")

print("\nTesting different parameter combinations...")

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

predictions = best_model.predict(
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

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n====================================")
print("OPTIMIZATION COMPLETED")
print("====================================")

print("\nBest Parameters:")

for parameter, value in grid_search.best_params_.items():
    print(
        parameter,
        ":",
        value
    )

print("\nBest Cross-Validation Score:")
print(
    round(
        grid_search.best_score_,
        4
    )
)

print("\nOptimized Model Performance:")

print(
    "MAE:",
    round(mae, 3)
)

print(
    "RMSE:",
    round(rmse, 3)
)

print(
    "R2 Score:",
    round(r2, 3)
)

joblib.dump(
    best_model,
    "models/optimized_model.pkl"
)

print("\nOptimized model saved at:")
print("models/optimized_model.pkl")