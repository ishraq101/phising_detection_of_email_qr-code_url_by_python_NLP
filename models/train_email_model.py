
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

data_file = Path("data/email_cleaned.csv")
model_file = Path("models/email_model.pkl")

print("Loading cleaned dataset...")
df = pd.read_csv(data_file, low_memory=False)

df = df.dropna(subset=["email_text", "label"])
df["email_text"] = df["email_text"].astype(str)
df["label"] = df["label"].astype(int)

X = df["email_text"]
y = df["label"]

print("Total emails:", len(df))
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training TF-IDF + Logistic Regression...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(X_train, y_train)

print("\nEvaluating model...")
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Benign (0)", "Phishing (1)"],
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nSaving model...")
joblib.dump(model, model_file)

print("Model saved to:", model_file)
