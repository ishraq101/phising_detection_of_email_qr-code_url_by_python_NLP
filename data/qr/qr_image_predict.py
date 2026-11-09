
from pathlib import Path
import re
import math
import joblib
import cv2
import pandas as pd

from collections import Counter
from urllib.parse import urlparse, parse_qsl
import tldextract


# ======================================
# PATH
# ======================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "qr" / "domain_ablation" / "xgboost.pkl"

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


# ======================================
# FEATURE EXTRACTION
# ======================================

def extract_features(url):

    parsed = urlparse(url)
    ext = extractor(url)

    hostname = parsed.hostname or ""
    domain = ext.domain or ""
    suffix = ext.suffix or ""
    subdomain = ext.subdomain or ""

    counts = Counter(url)
    entropy = -sum(
        (count / len(url)) * math.log2(count / len(url))
        for count in counts.values()
    )

    tokens = re.findall(r"[a-zA-Z0-9]+", url)

    suspicious_extensions = (
        ".exe", ".scr", ".bat", ".cmd", ".js",
        ".vbs", ".zip", ".rar", ".apk", ".msi"
    )

    return {
        "url_length": len(url),
        "dot_count": hostname.count("."),
        "url_entropy": entropy,
        "token_count": len(tokens),
        "subdomain_count": len(subdomain.split(".")) if subdomain else 0,
        "query_param_count": len(parse_qsl(parsed.query)),
        "tld_length": len(suffix),
        "has_hyphen_in_domain": int("-" in domain),
        "suspicious_file_extension": int(
            parsed.path.lower().endswith(suspicious_extensions)
        ),
        "domain_name_length": len(domain)
    }


# ======================================
# QR IMAGE PREDICTION
# ======================================

def predict_qr(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: QR image not found.")
        return

    detector = cv2.QRCodeDetector()
    url, points, _ = detector.detectAndDecode(image)

    if not url:
        print("Could not decode QR code.")
        return

    print("\nDecoded URL:", url)

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    features = extract_features(url)

    X = pd.DataFrame([features], columns=FEATURES)

    model = joblib.load(MODEL_PATH)

    prediction = int(model.predict(X)[0])
    probability = model.predict_proba(X)[0]

    print("\n========== QR PREDICTION ==========")

    if prediction == 0:
        print("Result: PHISHING")
    else:
        print("Result: LEGITIMATE")

    print("Phishing probability:", round(probability[0] * 100, 2), "%")
    print("Legitimate probability:", round(probability[1] * 100, 2), "%")


# ======================================
# RUN
# ======================================

if __name__ == "__main__":

    image_path = input("Enter QR image path: ").strip().strip('"')

    predict_qr(image_path)
