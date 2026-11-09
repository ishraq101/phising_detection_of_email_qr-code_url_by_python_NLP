
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

FILE = BASE_DIR / "processed" / "qr_features_cleaned.csv"

df = pd.read_csv(FILE)

print("\n========== PATH LENGTH BY CLASS ==========")

print(
    df.groupby("ClassLabel")["path_length"].describe()
)

print("\n========== UNIQUE PATH LENGTH VALUES ==========")

print(
    df.groupby("ClassLabel")["path_length"]
    .value_counts()
    .groupby(level=0)
    .head(15)
)

print("\n========== SAMPLE URLS ==========")

for label, class_name in [(0, "Phishing"), (1, "Legitimate")]:

    print(f"\n--- {class_name} samples ---")

    sample = df[df["ClassLabel"] == label][
        ["URL", "path_length", "url_length", "ClassLabel"]
    ].head(15)

    print(sample.to_string(index=False))
