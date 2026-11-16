
from pathlib import Path
import re
import math
import joblib
import cv2
import pandas as pd
import tldextract

from collections import Counter
from urllib.parse import urlparse, parse_qsl


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models" / "qr" / "consistent_features" / "xgboost.pkl"
)

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

extractor = tldextract.TLDExtract(suffix_list_urls=())

model = joblib.load(MODEL_PATH)


def extract_features(url):

    url = str(url).strip()

    if not url:
        raise ValueError("URL is empty.")

    # Keep the original URL for feature calculations.
    original_url = url

    # Add scheme only for URL parsing.
    parsing_url = url

    if not parsing_url.startswith(("http://", "https://")):
        parsing_url = "http://" + parsing_url

    parsed = urlparse(parsing_url)

    ext = extractor(parsing_url)

    hostname = parsed.hostname or ""
    domain = ext.domain or ""
    suffix = ext.suffix or ""
    subdomain = ext.subdomain or ""

    # URL entropy
    counts = Counter(original_url)
    total = len(original_url)

    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )

    # Token count
    tokens = re.findall(r"[a-zA-Z0-9]+", original_url)

    suspicious_extensions = (
        ".exe", ".scr", ".bat", ".cmd", ".js",
        ".vbs", ".zip", ".rar", ".apk", ".msi"
    )

    features = {
        "url_length": len(original_url),

        "dot_count": hostname.count("."),

        "url_entropy": entropy,

        "token_count": len(tokens),

        "subdomain_count": (
            len(subdomain.split(".")) if subdomain else 0
        ),

        "query_param_count": len(
            parse_qsl(parsed.query, keep_blank_values=True)
        ),

        "tld_length": len(suffix),

        "has_hyphen_in_domain": int("-" in hostname),

        "suspicious_file_extension": int(
            parsed.path.lower().endswith(suspicious_extensions)
        ),

        "domain_name_length": len(domain)
    }

    return features


def predict_url(url):

    features = extract_features(url)

    X = pd.DataFrame([features], columns=FEATURES)

    prediction = int(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    phishing_probability = probabilities[classes.index(0)]
    legitimate_probability = probabilities[classes.index(1)]

    return {
        "prediction": (
            "PHISHING" if prediction == 0 else "LEGITIMATE"
        ),

        "phishing_probability": round(
            float(phishing_probability) * 100, 2
        ),

        "legitimate_probability": round(
            float(legitimate_probability) * 100, 2
        )
    }


def predict_qr(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("QR image not found.")

    detector = cv2.QRCodeDetector()

    url, points, _ = detector.detectAndDecode(image)

    if not url:
        raise ValueError("Could not decode QR code.")

    result = predict_url(url)

    return url, result
