
import cv2
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "qr_images"
OUTPUT_FILE = BASE_DIR / "processed" / "qr_sample_decoding_results.csv"

detector = cv2.QRCodeDetector()

results = []

for label in ["phishing", "legitimate"]:
    folder = IMAGE_DIR / label

    for image_path in folder.glob("*.png"):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Cannot read: {image_path.name}")
            continue

        decoded_url, points, _ = detector.detectAndDecode(image)

        status = "Success" if decoded_url else "Failed"

        results.append({
            "image_name": image_path.name,
            "expected_label": label,
            "extracted_url": decoded_url,
            "decode_status": status
        })

        print(f"{image_path.name}: {status}")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_name",
            "expected_label",
            "extracted_url",
            "decode_status"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

successful = sum(
    1 for row in results
    if row["decode_status"] == "Success"
)

print("\n========== QR DECODING SUMMARY ==========")
print(f"Total images: {len(results)}")
print(f"Successfully decoded: {successful}")
print(f"Failed: {len(results) - successful}")
print(f"Results saved: {OUTPUT_FILE}")
