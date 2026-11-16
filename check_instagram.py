import pandas as pd

train = pd.read_csv("data/qr/processed/qr_train_domain.csv")

print("========== TOKEN COUNT DISTRIBUTION ==========")

print("\nClass 0 = Phishing")
print("Class 1 = Legitimate")

print(
    train.groupby("ClassLabel")["token_count"]
    .describe()
    .round(2)
)

print("\n========== TOKEN COUNT FREQUENCY ==========")

for label in [0, 1]:
    print(f"\nClass {label}:")

    print(
        train[train["ClassLabel"] == label]["token_count"]
        .value_counts()
        .sort_index()
        .head(20)
    )

print("\nInstagram token_count = 6")