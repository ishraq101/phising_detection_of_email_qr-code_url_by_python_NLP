
from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

from xgboost import XGBClassifier


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_FILE = BASE_DIR / "processed" / "qr_train_domain.csv"
TEST_FILE = BASE_DIR / "processed" / "qr_test_domain.csv"

MODEL_DIR = PROJECT_DIR / "models" / "qr" / "domain_split"
RESULT_DIR = BASE_DIR / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULT_DIR / "qr_model_comparison_domain.csv"


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading domain-wise QR datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nTraining class distribution:")
print(train_df["ClassLabel"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test_df["ClassLabel"].value_counts().sort_index())


# ==========================================
# 3. SELECT FEATURES AND LABEL
# ==========================================

TARGET = "ClassLabel"

# URL is not used as a direct model feature.
# Only extracted numerical features are used.

FEATURES = [
    col for col in train_df.columns
    if col not in ["URL", TARGET]
]

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)

# Ensure train and test have identical feature columns
X_test = X_test[FEATURES]

print("\nNumber of features:", len(FEATURES))
print("Features:", FEATURES)


# ==========================================
# 4. DEFINE MODELS
# ==========================================

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


# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Probability scores
    y_probability = model.predict_proba(X_test)

    # Class 0 = Phishing
    phishing_index = list(model.classes_).index(0)
    phishing_scores = y_probability[:, phishing_index]

    # Convert labels:
    # Phishing (0) = 1
    # Legitimate (1) = 0
    y_phishing = (y_test == 0).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        pos_label=0,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_phishing,
        phishing_scores
    )

    pr_auc = average_precision_score(
        y_phishing,
        phishing_scores
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\nAccuracy:", round(accuracy, 6))
    print("Phishing Precision:", round(precision, 6))
    print("Phishing Recall:", round(recall, 6))
    print("Phishing F1-score:", round(f1, 6))
    print("ROC-AUC:", round(roc_auc, 6))
    print("PR-AUC:", round(pr_auc, 6))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Phishing", "Legitimate"],
            zero_division=0
        )
    )

    # Save model separately
    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = MODEL_DIR / model_filename
    joblib.dump(model, model_path)

    print("Model saved:", model_path)

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Phishing_Precision": precision,
        "Phishing_Recall": recall,
        "Phishing_F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": cm[0][0],
        "FP": cm[0][1],
        "FN": cm[1][0],
        "TP": cm[1][1]
    })


# ==========================================
# 6. SAVE MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("DOMAIN-WISE MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nComparison saved:", OUTPUT_FILE)

print("\nDomain-wise model training completed successfully.")
