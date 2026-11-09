
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

train = pd.read_csv(
    BASE_DIR / "processed" / "qr_train.csv"
)

test = pd.read_csv(
    BASE_DIR / "processed" / "qr_test.csv"
)

def extract_domain(url):
    try:
        parsed = urlparse(url if "://" in url else "//" + url)
        return (parsed.hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""

train_domains = set(train["URL"].apply(extract_domain))
test_domains = set(test["URL"].apply(extract_domain))

overlap = train_domains.intersection(test_domains)

print("========== DOMAIN OVERLAP CHECK ==========")
print("Unique train domains:", len(train_domains))
print("Unique test domains:", len(test_domains))
print("Common domains:", len(overlap))
print(
    "Domain overlap percentage:",
    round(len(overlap) / max(len(test_domains), 1) * 100, 2),
    "%"
)

print("\nSample overlapping domains:")
for domain in list(sorted(overlap))[:20]:
    print(domain)
