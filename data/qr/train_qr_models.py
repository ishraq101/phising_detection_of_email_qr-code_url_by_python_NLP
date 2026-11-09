
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

TRAIN_FILE = BASE_DIR / "processed" / "qr_train.csv"
TEST_FILE = BASE_DIR / "processed" / "qr_test.csv"

MODEL_DIR = PROJECT_DIR / "models" / "qr"
RESULT_DIR = BASE_DIR / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=["URL", "ClassLabel"])
y_train = train_df["ClassLabel"]

X_test = test_df.drop(columns=["URL", "ClassLabel"])
y_test = test_df["ClassLabel"]

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Train and evaluate
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Class 0 = Phishing
    phishing_index = list(model.classes_).index(0)

    phishing_scores = probabilities[:, phishing_index]
    y_phishing = (y_test == 0).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label=0
    )

    roc_auc = roc_auc_score(y_phishing, phishing_scores)
    pr_auc = average_precision_score(y_phishing, phishing_scores)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Phishing Precision: {precision:.4f}")
    print(f"Phishing Recall: {recall:.4f}")
    print(f"Phishing F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"]
    ))

    # Save model
    model_filename = name.lower().replace(" ", "_")
    model_path = MODEL_DIR / f"{model_filename}.pkl"

    joblib.dump(model, model_path)

    # Save metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Phishing_Precision": precision,
        "Phishing_Recall": recall,
        "Phishing_F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    })

    print("Model saved:", model_path)

# Save comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "qr_model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nAll models trained successfully!")
