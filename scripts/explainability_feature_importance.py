from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score


DATA_PATH = Path("data/processed/heart_attack_cleaned.csv")
FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("Training explainable logistic regression model...")
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.75).astype(int)

print("\nClassification Report at threshold 0.75:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# -----------------------------------
# 1. Logistic Regression Coefficients
# -----------------------------------
classifier = model.named_steps["classifier"]

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": classifier.coef_[0]
})

coef_df["absolute_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("absolute_coefficient", ascending=False)

coef_df.to_csv(TABLE_DIR / "logistic_regression_feature_importance.csv", index=False)

top_coef = coef_df.head(15).sort_values("coefficient")

plt.figure(figsize=(10, 7))
plt.barh(top_coef["feature"], top_coef["coefficient"])
plt.title("Top Features by Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_logistic_regression_coefficients.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved logistic regression coefficient plot.")

# -----------------------------------
# 2. Permutation Importance
# -----------------------------------
print("\nCalculating permutation importance. This may take a few minutes...")

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
})

perm_df = perm_df.sort_values("importance_mean", ascending=False)
perm_df.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)

top_perm = perm_df.head(15).sort_values("importance_mean")

plt.figure(figsize=(10, 7))
plt.barh(top_perm["feature"], top_perm["importance_mean"])
plt.title("Top Features by Permutation Importance")
plt.xlabel("Mean Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_permutation_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved permutation importance plot.")

print("\nExplainability completed successfully.")
print("Tables saved in:", TABLE_DIR)
print("Figures saved in:", FIG_DIR)