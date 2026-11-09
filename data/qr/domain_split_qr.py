
import pandas as pd
import tldextract
import ipaddress

from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "processed" / "qr_features_cleaned.csv"

TRAIN_FILE = BASE_DIR / "processed" / "qr_train_domain.csv"
TEST_FILE = BASE_DIR / "processed" / "qr_test_domain.csv"

df = pd.read_csv(INPUT_FILE)

# Extract registered domain without downloading a suffix list
extractor = tldextract.TLDExtract(suffix_list_urls=())

def extract_registered_domain(url):
    url = str(url).strip().strip("'\"")

    if "://" not in url:
        url = "//" + url

    try:
        host = urlparse(url).hostname

        if not host:
            return ""

        host = host.lower().rstrip(".")

        # Handle IP addresses
        try:
            return str(ipaddress.ip_address(host))
        except ValueError:
            pass

        result = extractor(host)

        # Group subdomains under the same registered domain
        domain = result.top_domain_under_public_suffix

        return domain.lower() if domain else host

    except Exception:
        return ""

print("Extracting registered domains...")

df["domain_group"] = df["URL"].apply(
    extract_registered_domain
)

# Remove records with invalid domains
df = df[df["domain_group"] != ""].copy()

X = df.drop(columns=["URL", "ClassLabel", "domain_group"])
y = df["ClassLabel"]
groups = df["domain_group"]

# Five-fold stratified group split
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

train_df = df.iloc[train_idx].drop(
    columns=["domain_group"]
)

test_df = df.iloc[test_idx].drop(
    columns=["domain_group"]
)

# Verify domain separation
train_domains = set(groups.iloc[train_idx])
test_domains = set(groups.iloc[test_idx])

overlap = train_domains.intersection(test_domains)

# Save separate files; preserve original random split
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\n========== DOMAIN SPLIT SUMMARY ==========")

print("Total records:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining class distribution:")
print(train_df["ClassLabel"].value_counts())

print("\nTesting class distribution:")
print(test_df["ClassLabel"].value_counts())

print("\nUnique training domains:", len(train_domains))
print("Unique testing domains:", len(test_domains))
print("Common domains:", len(overlap))

print("\nDomain overlap percentage:",
      round(len(overlap) / max(len(test_domains), 1) * 100, 2),
      "%")

print("\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)
