
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "processed" / "qr_features_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

# Separate features and target
X = df.drop(columns=["URL", "ClassLabel"])
y = df["ClassLabel"]

# Stratified 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preserve URLs for later error analysis
train_df = pd.concat([
    df.loc[X_train.index, ["URL"]],
    X_train,
    y_train.rename("ClassLabel")
], axis=1)

test_df = pd.concat([
    df.loc[X_test.index, ["URL"]],
    X_test,
    y_test.rename("ClassLabel")
], axis=1)

train_df.to_csv(
    BASE_DIR / "processed" / "qr_train.csv",
    index=False
)

test_df.to_csv(
    BASE_DIR / "processed" / "qr_test.csv",
    index=False
)

print("========== DATA SPLIT SUMMARY ==========")
print("Total records:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nSaved: qr_train.csv and qr_test.csv")
