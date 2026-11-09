
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "processed" / "qr_features_cleaned.csv"

df = pd.read_csv(FILE)

print("\n========== CLASS DISTRIBUTION BY URL SOURCE ==========")

for label, name in [(0, "Phishing"), (1, "Legitimate")]:
    print(f"\n{name}:")
    print(
        df[df["ClassLabel"] == label]["URL"]
        .str.extract(r"https?://([^/:]+)")[0]
        .value_counts()
        .head(15)
    )

print("\n========== FEATURE STATISTICS BY CLASS ==========")

features = [
    "https_flag",
    "number_of_digits",
    "percentage_numeric_chars",
    "token_count",
    "has_ip_address",
    "dot_count",
    "tld_length",
    "tld_popularity",
    "domain_name_length",
    "url_length"
]

print(
    df.groupby("ClassLabel")[features]
    .mean()
    .round(3)
    .T
)

print("\n========== TOP DOMAINS BY CLASS ==========")

df["domain"] = df["URL"].str.extract(
    r"https?://([^/:]+)"
)[0].str.lower()

for label, name in [(0, "Phishing"), (1, "Legitimate")]:
    print(f"\n{name} domains:")
    print(
        df[df["ClassLabel"] == label]["domain"]
        .value_counts()
        .head(15)
    )
