
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv(
    "data/email_cleaned.csv",
    usecols=["subject", "body", "source", "label"],
    low_memory=False,
)

df["subject"] = df["subject"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)
df["email_text"] = (df["subject"] + " " + df["body"]).str.strip()
df["label"] = df["label"].astype(int)

df = df[df["email_text"] != ""].copy()

train = df[df["source"].isin(["trec5", "trec6"])]
test = df[df["source"] == "trec7"]

print("Training emails:", len(train))
print("Unseen-source test emails:", len(test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

print("\nTraining on trec5 + trec6...")
model.fit(train["email_text"], train["label"])

print("\nTesting on trec7...")
predictions = model.predict(test["email_text"])

print("Accuracy:", round(
    accuracy_score(test["label"], predictions), 4
))

print("\nClassification Report:")
print(classification_report(
    test["label"],
    predictions,
    target_names=["Benign (0)", "Phishing (1)"],
))
