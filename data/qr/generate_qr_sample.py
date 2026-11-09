import pandas as pd
import qrcode
from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "processed" / "qr_cleaned.csv"
PHISHING_DIR = BASE_DIR / "qr_images" / "phishing"
LEGITIMATE_DIR = BASE_DIR / "qr_images" / "legitimate"

PHISHING_DIR.mkdir(parents=True, exist_ok=True)
LEGITIMATE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Select 5 URLs from each class
sample = pd.concat([
    df[df["ClassLabel"] == 0].head(5),
    df[df["ClassLabel"] == 1].head(5)
])

for index, row in sample.iterrows():
    url = row["URL"]
    label = int(row["ClassLabel"])

    if label == 0:
        output_dir = PHISHING_DIR
        category = "phishing"
    else:
        output_dir = LEGITIMATE_DIR
        category = "legitimate"

    filename = f"{category}_{index}.png"

    qr = qrcode.make(url)
    qr.save(output_dir / filename)

print("QR sample generation completed!")
print("Phishing QR images:", len(list(PHISHING_DIR.glob("*.png"))))
print("Legitimate QR images:", len(list(LEGITIMATE_DIR.glob("*.png"))))