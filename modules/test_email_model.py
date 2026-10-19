
import joblib
from pathlib import Path

model_path = Path("models/email_model.pkl")

print("Loading email phishing detection model...")
model = joblib.load(model_path)

print("\nEmail Phishing Detector")
print("Type 'exit' to quit.")

while True:
    email_text = input("\nEnter email subject/body: ").strip()

    if email_text.lower() == "exit":
        print("Program closed.")
        break

    if not email_text:
        print("Please enter some email text.")
        continue

    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]

    confidence = probabilities[int(prediction)] * 100

    if prediction == 1:
        print("Prediction: Potential Phishing Email")
    else:
        print("Prediction: Potential Benign Email")

    print(f"Model confidence: {confidence:.2f}%")
