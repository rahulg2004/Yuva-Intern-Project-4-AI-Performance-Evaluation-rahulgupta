import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

data = pd.read_csv(
    "data/student_data.csv"
)

X = data.drop(
    "final_score",
    axis=1
)

model = joblib.load(
    "models/baseline_model.pkl"
)

importance = model.feature_importances_

result = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

result = result.sort_values(
    by="Importance",
    ascending=False
)

print("====================================")
print("FEATURE IMPORTANCE")
print("====================================")

print(result.to_string(index=False))

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    result["Feature"],
    result["Importance"]
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)

plt.title(
    "Feature Importance of Student Performance Model"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png"
)

plt.show()

print("\nFeature importance graph saved at:")
print("results/feature_importance.png")