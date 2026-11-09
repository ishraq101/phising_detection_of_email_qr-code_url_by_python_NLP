
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

TRAIN_FILE = BASE_DIR / "processed" / "qr_train_domain.csv"
TEST_FILE = BASE_DIR / "processed" / "qr_test_domain.csv"

MODEL_DIR = PROJECT_DIR / "models" / "qr" / "domain_ablation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = (
    BASE_DIR / "processed" /
    "qr_model_comparison_domain_ablation.csv"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading domain-wise QR datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# ==========================================
# 3. FEATURE SELECTION
# ==========================================

EXCLUDED_FEATURES = [
    "URL",
    "ClassLabel",
    "path_length",
    "https_flag",
    "has_ip_address",
    "number_of_digits",
    "percentage_numeric_chars",
    "tld_popularity"
]

FEATURES = [
    col for col in train_df.columns
    if col not in EXCLUDED_FEATURES
]

X_train = train_df[FEATURES]
y_train = train_df["ClassLabel"]

X_test = test_df[FEATURES]
y_test = test_df["ClassLabel"]

print("\nSelected features:")
print(FEATURES)

print("\nNumber of selected features:", len(FEATURES))


# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}


# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", model_name)
    print("=" * 50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 0]

    accuracy = accuracy_score(y_test, y_pred)

    # Class 0 = Phishing
    precision = precision_score(
        y_test, y_pred, pos_label=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label=0
    )

    roc_auc = roc_auc_score(
    (y_test == 0).astype(int), y_prob
)

    pr_auc = average_precision_score(
        (y_test == 0).astype(int), y_prob
    )

    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    print("Accuracy:", round(accuracy, 6))
    print("Phishing Precision:", round(precision, 6))
    print("Phishing Recall:", round(recall, 6))
    print("Phishing F1-score:", round(f1, 6))
    print("ROC-AUC:", round(roc_auc, 6))
    print("PR-AUC:", round(pr_auc, 6))

    print("\nConfusion Matrix:")
    print(cm)

    # Save trained model
    model_filename = (
        model_name.lower().replace(" ", "_") + ".pkl"
    )

    model_path = MODEL_DIR / model_filename

    joblib.dump(model, model_path)

    print("Model saved:", model_path)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Phishing_Precision": precision,
        "Phishing_Recall": recall,
        "Phishing_F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Phishing_Correct": cm[0][0],
        "Phishing_Missed": cm[0][1],
        "Legitimate_False_Alarm": cm[1][0],
        "Legitimate_Correct": cm[1][1]
    })


# ==========================================
# 6. SAVE COMPARISON RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(RESULT_FILE, index=False)

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

print(results_df.to_string(index=False))

print("\nResults saved:", RESULT_FILE)

print("\nFeature ablation experiment completed!")
