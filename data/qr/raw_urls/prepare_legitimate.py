import pandas as pd
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent
ZIP_FILE = BASE_DIR / "top-1m.csv.zip"

with zipfile.ZipFile(ZIP_FILE, "r") as archive:
    csv_name = archive.namelist()[0]

    with archive.open(csv_name) as file:
        df = pd.read_csv(
            file,
            header=None,
            names=["rank", "domain"],
            nrows=1000
        )

df["url"] = "https://" + df["domain"].astype(str) + "/"
df["label"] = 0
df["source"] = "Tranco"

output = BASE_DIR / "legitimate_urls.csv"
df.to_csv(output, index=False)

print("Dataset created successfully!")
print("Total URLs:", len(df))
print(df.head())