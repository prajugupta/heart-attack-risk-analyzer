from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("data/clean/heart_big_data.csv")

MODEL_DIR = Path("model")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading big dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["heart_attack"].value_counts())
print(df["heart_attack"].value_counts(normalize=True) * 100)

X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

# Convert all columns to numeric safely
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Fill missing values
for col in X.columns:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Reduce memory usage
X = X.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Balance training importance without changing test data
sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.06,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    random_state=42
)

print("\nTraining big model...")
model.fit(X_train, y_train, sample_weight=sample_weight)

print("\nPredicting probabilities...")
y_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.05)

results = []
best_score = -1
best_threshold = 0.50

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    row = {
        "model": "hist_gradient_boosting_big",
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    results.append(row)

    # Product score: not fake accuracy; balances F1, ROC-AUC, recall
    product_score = (0.40 * f1) + (0.35 * roc_auc) + (0.20 * recall) + (0.05 * precision)

    if product_score > best_score:
        best_score = product_score
        best_threshold = float(threshold)

results_df = pd.DataFrame(results)

results_df_by_accuracy = results_df.sort_values(
    by=["accuracy", "f1_score", "roc_auc"],
    ascending=False
)

results_df_by_product = results_df.sort_values(
    by=["f1_score", "recall", "roc_auc"],
    ascending=False
)

results_path = RESULT_DIR / "big_model_results.csv"
results_df.to_csv(results_path, index=False)

print("\nTop 15 by accuracy:")
print(results_df_by_accuracy.head(15).to_string(index=False))

print("\nTop 15 by F1/Recall:")
print(results_df_by_product.head(15).to_string(index=False))

print("\nBest selected threshold:", best_threshold)

# Final evaluation
final_pred = (y_proba >= best_threshold).astype(int)

print("\nFinal Product Model Report:")
print(classification_report(y_test, final_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(model, MODEL_DIR / "heart_model.pkl")

with open(MODEL_DIR / "threshold.txt", "w") as f:
    f.write(str(best_threshold))

with open(MODEL_DIR / "features.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

with open(MODEL_DIR / "model_info.txt", "w") as f:
    f.write("Model: HistGradientBoostingClassifier\n")
    f.write(f"Training file: {DATA_PATH}\n")
    f.write(f"Rows used: {len(df)}\n")
    f.write(f"Features used: {X.shape[1]}\n")
    f.write(f"Heart attack cases: {int(y.sum())}\n")
    f.write(f"No heart attack cases: {int((y == 0).sum())}\n")
    f.write(f"Best threshold: {best_threshold}\n")
    f.write(f"ROC-AUC: {roc_auc_score(y_test, y_proba)}\n")

print("\nSaved files:")
print(MODEL_DIR / "heart_model.pkl")
print(MODEL_DIR / "threshold.txt")
print(MODEL_DIR / "features.txt")
print(MODEL_DIR / "model_info.txt")
print(results_path)