import pandas as pd
import numpy as np
import os

np.random.seed(42)

os.makedirs("data", exist_ok=True)

number_of_students = 1000

study_hours = np.random.uniform(1, 10, number_of_students)

attendance = np.random.uniform(
    50,
    100,
    number_of_students
)

previous_score = np.random.uniform(
    40,
    95,
    number_of_students
)

assignments = np.random.randint(
    1,
    11,
    number_of_students
)

sleep_hours = np.random.uniform(
    4,
    9,
    number_of_students
)

participation = np.random.randint(
    1,
    11,
    number_of_students
)

final_score = (
    study_hours * 3
    + attendance * 0.25
    + previous_score * 0.35
    + assignments * 1.5
    + sleep_hours * 1
    + participation * 1.2
    + np.random.normal(0, 5, number_of_students)
)

final_score = np.clip(
    final_score,
    0,
    100
)

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignments": assignments,
    "sleep_hours": sleep_hours,
    "participation": participation,
    "final_score": final_score
})

data.to_csv(
    "data/student_data.csv",
    index=False
)

print("====================================")
print("DATASET CREATED SUCCESSFULLY")
print("====================================")

print("Number of students:", len(data))
print("Number of features:", len(data.columns) - 1)

print("\nFirst 5 records:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nDataset saved at:")
print("data/student_data.csv")