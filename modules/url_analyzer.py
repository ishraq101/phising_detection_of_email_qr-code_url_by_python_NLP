
from urllib.parse import urlparse
import ipaddress
import re

def analyze_url(url):

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    features = {}

    # 1. URL length
    features["url_length"] = len(url)

    # 2. HTTPS check
    features["uses_https"] = parsed.scheme == "https"

    # 3. IP address check
    try:
        ipaddress.ip_address(hostname)
        features["uses_ip_address"] = True
    except ValueError:
        features["uses_ip_address"] = False

    # 4. @ symbol check
    features["contains_at_symbol"] = "@" in url

    # 5. Hyphen in domain
    features["has_hyphen"] = "-" in hostname

    # 6. Subdomain count
    features["subdomain_count"] = max(0, len(hostname.split(".")) - 2)

    # 7. Suspicious keywords
    suspicious_words = [
        "login", "verify", "update", "secure",
        "account", "password", "bank", "confirm"
    ]

    found_words = [
        word for word in suspicious_words
        if word in url.lower()
    ]

    features["suspicious_keywords"] = found_words

    # 8. URL shortener check
    shorteners = [
        "bit.ly", "tinyurl.com", "t.co",
        "is.gd", "cutt.ly"
    ]

    features["uses_url_shortener"] = any(
        domain in hostname.lower() for domain in shorteners
    )

    # Display results
    print("\n========== URL ANALYSIS ==========")

    for feature, value in features.items():
        print(f"{feature}: {value}")

    print("==================================")

    return features


if __name__ == "__main__":

    url = input("Enter URL to analyze: ")

    analyze_url(url)
