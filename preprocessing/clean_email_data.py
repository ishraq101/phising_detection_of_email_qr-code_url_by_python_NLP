import pandas as pd
from pathlib import Path

input_file = Path("data/meajor_cleaned_preprocessed.csv")
output_file = Path("data/email_cleaned.csv")

print("Loading dataset...")

df = pd.read_csv(input_file, low_memory=False)

print("Original rows:", len(df))

# Remove rows without labels
df = df.dropna(subset=["label"]).copy()

# Keep only valid binary labels
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

# Combine email subject and body
df["subject"] = df["subject"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)
df["email_text"] = (
    df["subject"] + " " + df["body"]
).str.strip()

# Remove emails with no usable text
df = df[df["email_text"] != ""].copy()

df.to_csv(output_file, index=False)

print("\nCleaning completed!")
print("Cleaned rows:", len(df))
print("Label distribution:")
print(df["label"].value_counts().sort_index())
print("\nSaved to:", output_file)