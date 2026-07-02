from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

RAW_FILE = Path("data/raw/LLCP2024.XPT")
PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    raise FileNotFoundError(
        "LLCP2024.XPT not found. Put the extracted CDC BRFSS 2024 file here: data/raw/LLCP2024.XPT"
    )

target_col = "CVDINFR4"

features = [
    "_AGEG5YR",
    "_SEX",

    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",

    "_BMI5",

    "_TOTINDA",
    "_SMOKER3",
    "_RFDRHV9",

    "SLEPTIM1",

    "DIABETE4",
    "CVDSTRK3",
    "CHCKDNY2",
    "ASTHMA3",
    "CHCCOPD3",
    "ADDEPEV3",
    "HAVARTH4",

    "BPHIGH6",
    "TOLDHI3",

    "PRIMINS2",
    "PERSDOC3",
    "MEDCOST1",
    "CHECKUP1"
]

cols = [target_col] + features

print("Loading selected columns from BRFSS file...")
df, meta = pyreadstat.read_xport(str(RAW_FILE), usecols=cols)

print("Raw selected shape:", df.shape)

# Keep only valid target values
df = df[df[target_col].isin([1, 2])].copy()

# Target conversion
df["heart_attack"] = df[target_col].map({
    1: 1,   # Yes
    2: 0    # No
})

df = df.drop(columns=[target_col])

# Convert BMI: _BMI5 is BMI * 100
if "_BMI5" in df.columns:
    df["_BMI5"] = df["_BMI5"] / 100

# Column-wise missing-value cleaning
missing_codes = {
    "_AGEG5YR": [14],
    "_SEX": [],
    

    "GENHLTH": [7, 9],
    "PHYSHLTH": [77, 99],
    "MENTHLTH": [77, 99],

    "_BMI5": [99.99, 9999],
    "_TOTINDA": [9],
    "_SMOKER3": [9],
    "_RFDRHV9": [9],

    "SLEPTIM1": [77, 99],

    "DIABETE4": [7, 9],
    "CVDSTRK3": [7, 9],
    "CHCKDNY2": [7, 9],
    "ASTHMA3": [7, 9],
    "CHCCOPD3": [7, 9],
    "ADDEPEV3": [7, 9],
    "HAVARTH4": [7, 9],

    "BPHIGH6": [7, 9],
    "TOLDHI3": [7, 9],

    "PRIMINS2": [77, 99],
    "PERSDOC3": [7, 9],
    "MEDCOST1": [7, 9],
    "CHECKUP1": [7, 9]
}

for col, codes in missing_codes.items():
    if col in df.columns:
        df[col] = df[col].replace(codes, np.nan)

# Convert 88 in physical/mental health days to 0 days
for col in ["PHYSHLTH", "MENTHLTH"]:
    if col in df.columns:
        df[col] = df[col].replace(88, 0)

# Sleep should be 1 to 24 hours
if "SLEPTIM1" in df.columns:
    df.loc[(df["SLEPTIM1"] < 1) | (df["SLEPTIM1"] > 24), "SLEPTIM1"] = np.nan

# Impute missing values using median
for col in df.columns:
    if col != "heart_attack":
        df[col] = df[col].fillna(df[col].median())

# Save final cleaned file
output_path = PROCESSED_DIR / "heart_attack_cleaned.csv"
df.to_csv(output_path, index=False)

summary = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "heart_attack_no_count",
        "heart_attack_yes_count",
        "heart_attack_no_percent",
        "heart_attack_yes_percent"
    ],
    "value": [
        df.shape[0],
        df.shape[1],
        int((df["heart_attack"] == 0).sum()),
        int((df["heart_attack"] == 1).sum()),
        round(float((df["heart_attack"] == 0).mean() * 100), 2),
        round(float((df["heart_attack"] == 1).mean() * 100), 2)
    ]
})
summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

print("\nCleaned file saved at:", output_path)
print("Final shape:", df.shape)

print("\nTarget distribution:")
print(df["heart_attack"].value_counts())

print("\nTarget percentage:")
print(df["heart_attack"].value_counts(normalize=True) * 100)

print("\nDataset summary saved at:", TABLES_DIR / "dataset_summary.csv")
