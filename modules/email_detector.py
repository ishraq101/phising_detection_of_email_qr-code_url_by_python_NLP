
from pathlib import Path
import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "email_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_email(subject="", body=""):

    email_text = str(subject) + " " + str(body)
    email_text = email_text.strip()

    if not email_text:
        raise ValueError("Email content is empty.")

    prediction = int(model.predict([email_text])[0])
    probabilities = model.predict_proba([email_text])[0]

    classes = list(model.classes_)
    phishing_probability = probabilities[classes.index(1)]
    benign_probability = probabilities[classes.index(0)]

    return {
        "prediction": "PHISHING" if prediction == 1 else "BENIGN",
        "phishing_probability": round(
            float(phishing_probability) * 100, 2
        ),
        "benign_probability": round(
            float(benign_probability) * 100, 2
        )
    }


if __name__ == "__main__":

    print("========== EMAIL PHISHING DETECTOR ==========")

    subject = input("Enter email subject: ")
    body = input("Enter email body: ")

    result = predict_email(subject, body)

    print("\nPrediction:", result["prediction"])
    print("Phishing probability:", result["phishing_probability"], "%")
    print("Benign probability:", result["benign_probability"], "%")
