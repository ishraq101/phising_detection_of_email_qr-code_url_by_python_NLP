
import cv2

def scan_qr_code(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    detector = cv2.QRCodeDetector()

    data, points, _ = detector.detectAndDecode(image)

    if data:
        print("\nQR Code detected successfully!")
        print("Extracted content:", data)
    else:
        print("\nNo QR Code detected.")

if __name__ == "__main__":
    image_path = input("Enter QR image path: ")
    scan_qr_code(image_path)
