
import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qsl

CSV = Path(r"D:\Thesis\data\qr\raw_urls\url_features_extracted1.csv")

df = pd.read_csv(CSV).dropna(subset=["URL", "token_count"])

def candidates(url):
    return {
        "regex_alphanumeric": len(re.findall(r"[a-zA-Z0-9]+", url)),
        "regex_word": len(re.findall(r"\w+", url)),
        "split_dot": len(url.split(".")),
        "split_slash": len(url.split("/")),
        "split_all_separators": len([
            x for x in re.split(r"[./:_?&=-]+", url) if x
        ]),
        "split_non_alphanumeric": len([
            x for x in re.split(r"[^a-zA-Z0-9]+", url) if x
        ]),
    }

match_counts = {
    name: 0 for name in candidates(df.iloc[0]["URL"])
}

for _, row in df.head(1000).iterrows():
    actual = int(row["token_count"])
    results = candidates(str(row["URL"]))

    for name, value in results.items():
        if value == actual:
            match_counts[name] += 1

print("========== TOKEN COUNT ANALYSIS ==========")
print("Samples checked:", min(1000, len(df)))

for name, count in sorted(
    match_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {count} exact matches")

print("\n========== SAMPLE COMPARISON ==========")

for _, row in df.head(10).iterrows():
    print("\nURL:", row["URL"])
    print("Dataset token_count:", row["token_count"])
    print("Candidates:", candidates(str(row["URL"])))

print("\nAnalysis completed.")
