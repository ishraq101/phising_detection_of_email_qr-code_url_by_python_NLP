
import pandas as pd
import shap

from modules.qr_detector import extract_features, model

url = "https://www.instagram.com/__rahat__101_/"

# Extract URL features
features = extract_features(url)
X = pd.DataFrame([features])

# Ensure correct feature order
X = X[model.get_booster().feature_names]

# Get prediction
probabilities = model.predict_proba(X)[0]
classes = list(model.classes_)

phishing_prob = probabilities[classes.index(0)] * 100
legitimate_prob = probabilities[classes.index(1)] * 100

print("========== MODEL PREDICTION ==========")
print(f"Phishing Probability: {phishing_prob:.2f}%")
print(f"Legitimate Probability: {legitimate_prob:.2f}%")

# SHAP explanation
print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Binary XGBoost: explain class 1 (Legitimate)
if isinstance(shap_values, list):
    values = shap_values[1][0]
else:
    values = shap_values[0]

print("\n========== SHAP FEATURE CONTRIBUTIONS ==========")
print("Negative contribution = pushes toward PHISHING")
print("Positive contribution = pushes toward LEGITIMATE")

results = []

for name, value in zip(X.columns, values):
    results.append({
        "Feature": name,
        "Feature Value": features[name],
        "SHAP Value": float(value)
    })

results.sort(key=lambda x: abs(x["SHAP Value"]), reverse=True)

for item in results:
    print(
        f"{item['Feature']}: "
        f"Value={item['Feature Value']}, "
        f"SHAP={item['SHAP Value']:.6f}"
    )

print("\n========== EXPLANATION COMPLETED ==========")
