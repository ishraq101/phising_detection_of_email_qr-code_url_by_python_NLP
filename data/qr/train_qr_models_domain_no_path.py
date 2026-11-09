
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
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# PATHS
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_FILE = BASE_DIR / "processed" / "qr_train_domain.csv"
TEST_FILE = BASE_DIR / "processed" / "qr_test_domain.csv"

MODEL_DIR = PROJECT_DIR / "models" / "qr" / "domain_no_path"
RESULT_FILE = BASE_DIR / "processed" / "qr_model_comparison_domain_no_path.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# LOAD DATA
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

TARGET = "ClassLabel"

# Exclude URL, label and path_length
FEATURES = [
    col for col in train_df.columns
    if col not in ["URL", TARGET, "path_length"]
]

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Features:", FEATURES)


# MODELS
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


# TRAIN AND EVALUATE
results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    phishing_index = list(model.classes_).index(0)
    phishing_scores = probabilities[:, phishing_index]

    y_phishing = (y_test == 0).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=0)
    recall = recall_score(y_test, predictions, pos_label=0)
    f1 = f1_score(y_test, predictions, pos_label=0)

    roc_auc = roc_auc_score(y_phishing, phishing_scores)
    pr_auc = average_precision_score(y_phishing, phishing_scores)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("Accuracy:", round(accuracy, 6))
    print("Phishing Precision:", round(precision, 6))
    print("Phishing Recall:", round(recall, 6))
    print("Phishing F1:", round(f1, 6))
    print("ROC-AUC:", round(roc_auc, 6))
    print("PR-AUC:", round(pr_auc, 6))
    print("Confusion Matrix:\n", cm)

    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, MODEL_DIR / filename)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Phishing_Precision": precision,
        "Phishing_Recall": recall,
        "Phishing_F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Phishing_Correct": int(cm[0][0]),
        "Phishing_Missed": int(cm[0][1]),
        "Legitimate_False_Alarm": int(cm[1][0]),
        "Legitimate_Correct": int(cm[1][1])
    })


# SAVE RESULTS
results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_FILE, index=False)

print("\n========== FINAL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nModels saved in:", MODEL_DIR)
print("Results saved in:", RESULT_FILE)
