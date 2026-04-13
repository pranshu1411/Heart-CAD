import pandas as pd
from sklearn.preprocessing import StandardScaler

columns = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]

# Process Cleveland
cleveland = pd.read_csv("../original-data/cleveland.csv", header=None, names=columns)
cleveland = cleveland.replace("?", pd.NA)
cleveland = cleveland.apply(pd.to_numeric)
cleveland["target"] = cleveland["target"].apply(lambda x: 0 if x == 0 else 1)
cleveland = cleveland.drop(columns=["thal"])
cleveland = cleveland.dropna()

# Process Statlog
statlog = pd.read_csv("../original-data/statlog.csv", header=None, sep=r'\s+', names=columns)
statlog = statlog.replace("?", pd.NA)
statlog = statlog.apply(pd.to_numeric)
# Statlog uses 1 (absence) and 2 (presence), differing from Cleveland
statlog["target"] = statlog["target"].apply(lambda x: 0 if x == 1 else 1)
statlog = statlog.drop(columns=["thal"])
statlog = statlog.dropna()

indian = pd.read_csv("../original-data/Cardiovascular_Disease_Dataset.csv")

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

if "patientid" in indian.columns:
    indian = indian.drop(columns=["patientid"])

features = [
    "age","sex","cp","trestbps","chol",
    "thalach","exang","oldpeak","slope","ca"
]

cleveland = cleveland[features + ["target"]]
indian = indian[features + ["target"]]
statlog = statlog[features + ["target"]]

chol_cap = cleveland['chol'].quantile(0.99)
cleveland['chol'] = cleveland['chol'].clip(upper=chol_cap)
indian['chol'] = indian['chol'].clip(upper=chol_cap)
statlog['chol'] = statlog['chol'].clip(upper=chol_cap)

cleveland.to_csv("../processed-data/cleveland_processed.csv", index=False)
indian.to_csv("../processed-data/indian_processed.csv", index=False)
statlog.to_csv("../processed-data/statlog_processed.csv", index=False)

continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

scaler = StandardScaler()
scaler.fit(cleveland[continuous_features])

cleveland_norm = cleveland.copy()
indian_norm = indian.copy()
statlog_norm = statlog.copy()

cleveland_norm[continuous_features] = scaler.transform(cleveland[continuous_features])
indian_norm[continuous_features] = scaler.transform(indian[continuous_features])
statlog_norm[continuous_features] = scaler.transform(statlog[continuous_features])

cleveland_norm.to_csv("../processed-data/cleveland_normalised.csv", index=False)
indian_norm.to_csv("../processed-data/indian_normalised.csv", index=False)
statlog_norm.to_csv("../processed-data/statlog_normalised.csv", index=False)

print("Processing complete")
print("Cleveland shape:", cleveland.shape)
print("Indian shape:", indian.shape)
print("Statlog shape:", statlog.shape)
print("\nNormalisation complete (StandardScaler on continuous features)")
print("Normalised files saved: cleveland_normalised.csv, indian_normalised.csv, statlog_normalised.csv")