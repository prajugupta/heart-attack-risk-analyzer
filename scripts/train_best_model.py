from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

DATA_PATH = Path("data/processed/heart_attack_cleaned.csv")
MODEL_DIR = Path("outputs/models")
TABLE_DIR = Path("outputs/tables")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading full cleaned dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df["heart_attack"].value_counts())
print(df["heart_attack"].value_counts(normalize=True) * 100)

X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "random_forest_balanced": RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "extra_trees_balanced": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
best_model = None
best_model_name = None
best_score = -1
best_threshold = 0.50

for name, model in models.items():
    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Try different thresholds
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        y_pred = (y_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            "model": name,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        # For product: prioritize F1 + ROC-AUC, not fake accuracy
        product_score = (0.45 * f1) + (0.35 * roc_auc) + (0.20 * recall)

        if product_score > best_score:
            best_score = product_score
            best_model = model
            best_model_name = name
            best_threshold = threshold

    print("Done:", name)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=["f1_score", "roc_auc", "recall"],
    ascending=False
)

results_path = TABLE_DIR / "best_model_training_results.csv"
results_df.to_csv(results_path, index=False)

print("\nTop 10 model results:")
print(results_df.head(10).to_string(index=False))

print("\nBest selected model:", best_model_name)
print("Best threshold:", best_threshold)

# Save best model
model_path = MODEL_DIR / "heart_attack_best_model.pkl"
threshold_path = MODEL_DIR / "best_threshold.txt"
features_path = MODEL_DIR / "model_features.txt"

joblib.dump(best_model, model_path)

with open(threshold_path, "w") as f:
    f.write(str(best_threshold))

with open(features_path, "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print("\nSaved best model at:", model_path)
print("Saved threshold at:", threshold_path)
print("Saved feature list at:", features_path)

# Final report for best model
best_y_proba = best_model.predict_proba(X_test)[:, 1]
best_y_pred = (best_y_proba >= best_threshold).astype(int)

print("\nFinal Best Model Classification Report:")
print(classification_report(y_test, best_y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, best_y_proba))
print("\nTraining complete.")