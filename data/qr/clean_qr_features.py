
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "raw_urls" / "url_features_extracted1.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "qr_features_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

print("Original dataset:", df.shape)

# Remove records with missing labels
df = df.dropna(subset=["ClassLabel"])

# Check label consistency for duplicate URLs
conflicts = df.groupby("URL")["ClassLabel"].nunique()
conflicts = conflicts[conflicts > 1]

if not conflicts.empty:
    raise ValueError(
        f"Conflicting labels found for {len(conflicts)} URLs."
    )

# Remove duplicate URLs
df = df.drop_duplicates(subset=["URL"], keep="first")

# Validate labels
if not set(df["ClassLabel"].unique()).issubset({0, 1, 0.0, 1.0}):
    raise ValueError("Unexpected class labels found.")

df["ClassLabel"] = df["ClassLabel"].astype(int)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n========== CLEANING SUMMARY ==========")
print("Final shape:", df.shape)
print("\nClass distribution:")
print(df["ClassLabel"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate URLs:", df["URL"].duplicated().sum())
print("Saved to:", OUTPUT_FILE)
