
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "processed" / "qr_train_domain.csv"
TEST_FILE = BASE_DIR / "processed" / "qr_test_domain.csv"

# ==========================================
# 2. LOAD DATA
# ==========================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

TARGET = "ClassLabel"

FEATURES = [
    col for col in train_df.columns
    if col not in ["URL", TARGET]
]

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)

# ==========================================
# 3. FEATURE CORRELATION
# ==========================================

print("\n========== FEATURE CORRELATION ==========")

correlation = train_df[FEATURES + [TARGET]].corr(numeric_only=True)

label_corr = correlation[TARGET].drop(TARGET)

print("\nCorrelation with ClassLabel:")
print(label_corr.abs().sort_values(ascending=False))

# ==========================================
# 4. SINGLE FEATURE TEST
# ==========================================

print("\n========== SINGLE FEATURE ANALYSIS ==========")

single_feature_results = []

for feature in FEATURES:

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    model.fit(X_train[[feature]], y_train)

    predictions = model.predict(X_test[[feature]])

    accuracy = accuracy_score(y_test, predictions)

    single_feature_results.append({
        "Feature": feature,
        "Accuracy": accuracy
    })

single_df = pd.DataFrame(single_feature_results)

single_df = single_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nSingle-feature prediction accuracy:")
print(single_df.to_string(index=False))

# ==========================================
# 5. FEATURE IMPORTANCE
# ==========================================

print("\n========== FEATURE IMPORTANCE ==========")

importance_model = DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)

importance_model.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

# ==========================================
# 6. SAVE RESULTS
# ==========================================

RESULT_DIR = BASE_DIR / "processed"

single_df.to_csv(
    RESULT_DIR / "qr_single_feature_analysis.csv",
    index=False
)

importance_df.to_csv(
    RESULT_DIR / "qr_feature_importance.csv",
    index=False
)

print("\nAnalysis files saved successfully.")
