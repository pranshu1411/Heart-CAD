import pandas as pd

# Load Cleveland dataset

columns = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]

cleveland = pd.read_csv("original-data/cleveland.csv", header=None, names=columns)

# Convert '?' to NaN
cleveland = cleveland.replace("?", pd.NA)

# Convert all columns to numeric
cleveland = cleveland.apply(pd.to_numeric)

# Convert target from 0–4 → binary
cleveland["target"] = cleveland["target"].apply(lambda x: 0 if x == 0 else 1)

# Remove feature not present in Indian dataset
cleveland = cleveland.drop(columns=["thal"])

# Drop rows with missing values
cleveland = cleveland.dropna()

# Load Indian dataset

indian = pd.read_csv("original-data/Cardiovascular_Disease_Dataset.csv")

# Rename columns to match Cleveland dataset
indian = indian.rename(columns={
    "gender": "sex",
    "chestpain": "cp",
    "restingBP": "trestbps",
    "serumcholestrol": "chol",
    "fastingbloodsugar": "fbs",
    "restingrelectro": "restecg",
    "maxheartrate": "thalach",
    "exerciseangia": "exang",
    "noofmajorvessels": "ca"
})

# Remove patient ID
if "patientid" in indian.columns:
    indian = indian.drop(columns=["patientid"])

# Keep only common features

features = [
    "age","sex","cp","trestbps","chol",
    "thalach","exang","oldpeak","slope","ca"
]

cleveland = cleveland[features + ["target"]]
indian = indian[features + ["target"]]

# Save processed datasets

cleveland.to_csv("processed-data/cleveland_processed.csv", index=False)
indian.to_csv("processed-data/indian_processed.csv", index=False)


print("Processing complete")
print("Cleveland shape:", cleveland.shape)
print("Indian shape:", indian.shape)