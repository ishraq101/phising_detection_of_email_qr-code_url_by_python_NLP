
def hybrid_fusion(email_result=None, qr_result=None):

    if email_result is None and qr_result is None:
        raise ValueError("At least one detection result is required.")

    # Email only
    if qr_result is None:
        return {
            "final_prediction": email_result["prediction"],
            "email_probability": email_result["phishing_probability"],
            "fusion_method": "Email Only"
        }

    # QR only
    if email_result is None:
        return {
            "final_prediction": qr_result["prediction"],
            "qr_probability": qr_result["phishing_probability"],
            "fusion_method": "QR Only"
        }

    # Get phishing probabilities
    email_probability = float(email_result["phishing_probability"])
    qr_probability = float(qr_result["phishing_probability"])

    # Conservative OR-based decision
    if email_probability >= 50 or qr_probability >= 50:
        final_prediction = "PHISHING"
    else:
        final_prediction = "BENIGN"

    return {
        "final_prediction": final_prediction,
        "email_probability": round(email_probability, 2),
        "qr_probability": round(qr_probability, 2),
        "fusion_method": "Conservative OR-Based Decision-Level Fusion"
    }
