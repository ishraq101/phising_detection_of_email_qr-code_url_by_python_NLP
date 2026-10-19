
import pandas as pd

file_path = "data/email_cleaned.csv"

df = pd.read_csv(
    file_path,
    usecols=["subject", "body", "source", "label"],
    low_memory=False,
)

df["subject"] = df["subject"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)
df["label"] = df["label"].astype(int)

print("Email source distribution by label:")
print(pd.crosstab(df["source"], df["label"]))

print("\nEmail length statistics by label:")
df["email_length"] = (
    df["subject"].str.len() + df["body"].str.len()
)

print(
    df.groupby("label")["email_length"]
    .describe()
    .round(1)
)

print("\nSample benign emails:")
for text in df.loc[df["label"] == 0, "body"].head(3):
    print("\n---")
    print(text[:500])

print("\nSample phishing emails:")
for text in df.loc[df["label"] == 1, "body"].head(3):
    print("\n---")
    print(text[:500])
