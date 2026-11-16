
from modules.email_detector import predict_email
from modules.qr_detector import predict_url, predict_qr
from fusion.hybrid_fusion import hybrid_fusion


def show_result(result):

    print("\n========== PREDICTION RESULT ==========")

    for key, value in result.items():
        print(f"{key}: {value}")


def main():

    while True:

        print("\n===================================")
        print(" MULTIMODAL PHISHING DETECTION SYSTEM")
        print("===================================")

        print("1. Email Detection")
        print("2. QR Image Detection")
        print("3. URL Detection")
        print("4. Email + QR Hybrid Detection")
        print("5. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":

            subject = input("Email Subject: ")
            body = input("Email Body: ")

            result = predict_email(subject, body)
            show_result(result)

        elif choice == "2":

            image_path = input("Enter QR image path: ").strip().strip('"')

            try:
                url, result = predict_qr(image_path)

                print("\nDecoded URL:", url)
                show_result(result)

            except Exception as error:
                print("Error:", error)

        elif choice == "3":

            url = input("Enter URL: ").strip()

            try:
                result = predict_url(url)
                show_result(result)

            except Exception as error:
                print("Error:", error)

        elif choice == "4":

            print("\nEnter Email Information")
            subject = input("Email Subject: ")
            body = input("Email Body: ")

            image_path = input(
                "Enter QR image path: "
            ).strip().strip('"')

            try:
                email_result = predict_email(subject, body)

                url, qr_result = predict_qr(image_path)

                print("\nDecoded QR URL:", url)

                final_result = hybrid_fusion(
                    email_result,
                    qr_result
                )

                show_result(final_result)

            except Exception as error:
                print("Error:", error)

        elif choice == "5":

            print("Application closed.")
            break

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
