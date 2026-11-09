
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "raw_urls" / "url_features_extracted1.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "qr_cleaned.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))

# Remove rows with missing labels
df = df.dropna(subset=["ClassLabel"])

# Remove duplicate URLs
df = df.drop_duplicates(subset=["URL"])

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning completed!")
print("Final rows:", len(df))
print("\nClass distribution:")
print(df["ClassLabel"].value_counts())
print("\nSaved to:", OUTPUT_FILE)
