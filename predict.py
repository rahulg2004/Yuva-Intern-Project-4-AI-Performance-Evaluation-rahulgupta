import pandas as pd
import joblib

model = joblib.load(
    "models/optimized_model.pkl"
)

print("====================================")
print("STUDENT PERFORMANCE PREDICTOR")
print("====================================")

study_hours = float(
    input("Enter study hours per day: ")
)

attendance = float(
    input("Enter attendance percentage: ")
)

previous_score = float(
    input("Enter previous score: ")
)

assignments = int(
    input("Enter completed assignments out of 10: ")
)

sleep_hours = float(
    input("Enter average sleep hours: ")
)

participation = int(
    input("Enter participation level out of 10: ")
)

student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "assignments": [assignments],
    "sleep_hours": [sleep_hours],
    "participation": [participation]
})

prediction = model.predict(
    student
)

score = prediction[0]

score = max(
    0,
    min(
        100,
        score
    )
)

print("\n====================================")
print("PREDICTION RESULT")
print("====================================")

print(
    "Predicted Final Score:",
    round(score, 2)
)

if score >= 90:
    print("Performance Level: Excellent")

elif score >= 75:
    print("Performance Level: Very Good")

elif score >= 60:
    print("Performance Level: Good")

elif score >= 40:
    print("Performance Level: Average")

else:
    print("Performance Level: Needs Improvement")