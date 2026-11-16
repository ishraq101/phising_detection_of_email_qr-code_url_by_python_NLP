
from pathlib import Path
import re
import math
import joblib
import pandas as pd
import tldextract

from collections import Counter
from urllib.parse import urlparse, parse_qsl

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =====================================
# 1. PATHS
# =====================================

ROOT = Path(__file__).resolve().parent

TRAIN_FILE = ROOT / "data/qr/processed/qr_train_domain.csv"
TEST_FILE = ROOT / "data/qr/processed/qr_test_domain.csv"

MODEL_DIR = ROOT / "models/qr/consistent_features"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

extractor = tldextract.TLDExtract(suffix_list_urls=())

FEATURES = [
    "url_length",
    "dot_count",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "has_hyphen_in_domain",
    "suspicious_file_extension",
    "domain_name_length"
]

SUSPICIOUS_EXTENSIONS = (
    ".exe", ".scr", ".bat", ".cmd", ".js",
    ".vbs", ".zip", ".rar", ".apk", ".msi"
)


# =====================================
# 2. FEATURE EXTRACTION
# =====================================

def extract_features(url):

    url = str(url).strip()

    parsed_url = url

    if not parsed_url.startswith(("http://", "https://")):
        parsed_url = "http://" + parsed_url

    parsed = urlparse(parsed_url)
    ext = extractor(parsed_url)

    hostname = parsed.hostname or ""
    domain = ext.domain or ""
    suffix = ext.suffix or ""
    subdomain = ext.subdomain or ""

    total = len(url)

    if total == 0:
        raise ValueError("Empty URL")

    counts = Counter(url)

    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )

    tokens = re.findall(r"[a-zA-Z0-9]+", url)

    query_count = len(
        parse_qsl(parsed.query, keep_blank_values=True)
    )

    return {
        "url_length": len(url),
        "dot_count": hostname.count("."),
        "url_entropy": entropy,
        "token_count": len(tokens),
        "subdomain_count": (
            len(subdomain.split(".")) if subdomain else 0
        ),
        "query_param_count": query_count,
        "tld_length": len(suffix),
        "has_hyphen_in_domain": int("-" in hostname),
        "suspicious_file_extension": int(
            parsed.path.lower().endswith(SUSPICIOUS_EXTENSIONS)
        ),
        "domain_name_length": len(domain)
    }


# =====================================
# 3. LOAD DOMAIN-WISE DATA
# =====================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# =====================================
# 4. RE-EXTRACT FEATURES
# =====================================

print("\nExtracting training features...")

X_train = pd.DataFrame(
    [extract_features(url) for url in train_df["URL"]],
    columns=FEATURES
)

print("Extracting testing features...")

X_test = pd.DataFrame(
    [extract_features(url) for url in test_df["URL"]],
    columns=FEATURES
)

y_train = train_df["ClassLabel"].astype(int)
y_test = test_df["ClassLabel"].astype(int)

print("\nFeature extraction completed.")
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)


# =====================================
# 5. TRAIN XGBOOST
# =====================================

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model.fit(X_train, y_train)


# =====================================
# 6. EVALUATION
# =====================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

classes = list(model.classes_)
phishing_index = classes.index(0)

phishing_probabilities = probabilities[:, phishing_index]

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

roc_auc = roc_auc_score(
    (y_test == 0).astype(int),
    phishing_probabilities
)

pr_auc = average_precision_score(
    (y_test == 0).astype(int),
    phishing_probabilities
)

cm = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

print("\n========== CONSISTENT QR MODEL RESULTS ==========")

print("Accuracy:", round(accuracy, 6))
print("Phishing Precision:", round(precision, 6))
print("Phishing Recall:", round(recall, 6))
print("Phishing F1:", round(f1, 6))
print("ROC-AUC:", round(roc_auc, 6))
print("PR-AUC:", round(pr_auc, 6))

print("\nConfusion Matrix:")
print(cm)


# =====================================
# 7. SAVE MODEL
# =====================================

model_path = MODEL_DIR / "xgboost.pkl"

joblib.dump(model, model_path)

print("\nModel saved:", model_path)


# Save feature names
feature_path = MODEL_DIR / "features.txt"

with open(feature_path, "w") as file:
    file.write("\n".join(FEATURES))


# Save evaluation results
results = pd.DataFrame([{
    "Model": "XGBoost",
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
}])

results.to_csv(
    MODEL_DIR / "evaluation_results.csv",
    index=False
)

print("\nEvaluation saved.")
print("\nQR consistent-feature training completed!")
