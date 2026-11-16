
import pandas as pd
from modules.qr_detector import extract_features, FEATURES

CSV_PATH = (
    r"D:\Thesis\data\qr\raw_urls\url_features_extracted1.csv"
)

df = pd.read_csv(CSV_PATH)

print("========== QR FEATURE VERIFICATION ==========")

for index, row in df.head(5).iterrows():

    original = row[FEATURES].astype(float)
    extracted = extract_features(row["URL"])

    print(f"\nSample {index + 1}")
    print("URL:", row["URL"])

    mismatch = False

    for feature in FEATURES:
        old_value = float(original[feature])
        new_value = float(extracted[feature])

        if abs(old_value - new_value) > 0.00001:
            mismatch = True
            print(
                f"MISMATCH | {feature}: "
                f"Dataset={old_value}, Current={new_value}"
            )

    if not mismatch:
        print("All 10 features MATCH.")

print("\n========== VERIFICATION FINISHED ==========")
