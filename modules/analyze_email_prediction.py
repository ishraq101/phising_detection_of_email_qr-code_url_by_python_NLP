
import joblib
import numpy as np
from pathlib import Path

model_path = Path("models/email_model.pkl")
model = joblib.load(model_path)

email_text = """
We are pleased to inform you that the booklet on
Examination Rules, Regulations and Policies for Students
has been updated. The booklet is intended to help students
understand the university's examination-related rules,
regulations, and policies effective from Fall-2026 Semester.

All students are requested to read the updated booklet
carefully and familiarize themselves with the applicable
rules and regulations to ensure compliance with the
university's examination requirements.

The booklet is available on the University's official
website at the following link:

Examination Rules, Regulations and Policies for Students:
https://webbackend.daffodilvarsity.edu.bd/images/office-site/menus/Exam-rules-and-policies_for-Students.pdf

Your attention and cooperation in this regard are highly appreciated.
"""

# Get prediction and probabilities
prediction = model.predict([email_text])[0]
probabilities = model.predict_proba([email_text])[0]

print("Prediction:", "Phishing" if prediction == 1 else "Benign")
print(f"Benign probability: {probabilities[0] * 100:.2f}%")
print(f"Phishing probability: {probabilities[1] * 100:.2f}%")

# Inspect words contributing to the prediction
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

features = vectorizer.transform([email_text])
coefficients = classifier.coef_[0]

row = features.getrow(0)
contributions = row.data * coefficients[row.indices]

terms = vectorizer.get_feature_names_out()
indices = row.indices

positive_order = np.argsort(contributions)[::-1]
negative_order = np.argsort(contributions)

print("\nWords pushing toward PHISHING:")
shown = 0
for i in positive_order:
    if contributions[i] <= 0:
        break
    print(f"{terms[indices[i]]}: {contributions[i]:.4f}")
    shown += 1
    if shown == 10:
        break

print("\nWords pushing toward BENIGN:")
shown = 0
for i in negative_order:
    if contributions[i] >= 0:
        break
    print(f"{terms[indices[i]]}: {contributions[i]:.4f}")
    shown += 1
    if shown == 10:
        break
