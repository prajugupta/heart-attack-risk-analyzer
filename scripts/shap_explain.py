from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Paths
DATA_PATH = Path("data/clean/heart_big_data.csv")
MODEL_PATH = Path("model/heart_model.pkl")
FEATURES_PATH = Path("model/features.txt")

# Fallback paths if old structure is still used
if not DATA_PATH.exists():
    DATA_PATH = Path("data/processed/heart_attack_cleaned.csv")

if not MODEL_PATH.exists():
    MODEL_PATH = Path("outputs/models/heart_attack_best_model.pkl")

FIG_DIR = Path("graphs")
RESULT_DIR = Path("results")

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data and model...")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

# Use same features used during training
if FEATURES_PATH.exists():
    features = [line.strip() for line in FEATURES_PATH.read_text().splitlines() if line.strip()]
    X = X[features]

# Use sample for faster SHAP calculation
X_background = X.sample(n=min(100, len(X)), random_state=42)
X_sample = X.sample(n=min(500, len(X)), random_state=7)

print("Creating SHAP explainer...")

def predict_risk(data):
    data_df = pd.DataFrame(data, columns=X.columns)
    return model.predict_proba(data_df)[:, 1]

explainer = shap.Explainer(predict_risk, X_background)
shap_values = explainer(X_sample)

print("Generating SHAP bar plot...")

plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.savefig(FIG_DIR / "shap_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("Generating SHAP beeswarm plot...")

plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

print("Generating SHAP waterfall plot for one prediction...")

plt.figure()
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.savefig(FIG_DIR / "shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()

# Save mean absolute SHAP importance
importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap_value": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values("mean_abs_shap_value", ascending=False)
importance.to_csv(RESULT_DIR / "shap_features.csv", index=False)

print("\nSHAP explainability completed.")
print("Saved:")
print(FIG_DIR / "shap_bar.png")
print(FIG_DIR / "shap_beeswarm.png")
print(FIG_DIR / "shap_waterfall.png")
print(RESULT_DIR / "shap_features.csv")