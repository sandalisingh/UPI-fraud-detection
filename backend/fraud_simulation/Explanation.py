# Explanation.py
from backend.fraud_simulation.Network import MODEL_PATH
from backend import get_scaler, get_vectorizer, get_model

# RULE-BASED REASON ENGINE
def generate_reasons(x):
    reasons = []

    if x.get("Amount", 0) > 100000:
        reasons.append("Transaction amount exceeds typical limits")

    if x.get("Amount",0) > x.get("Avg_Transaction_Value", 1) * 10:
        reasons.append("Transaction amount is unusually high compared to average transaction value")
    
    # ---- Collect Request Scam ----
    if x.get("Transaction_Type", "") == "Collect_Request":
        reasons.append("Transaction type is 'Collect Request', which is commonly exploited in scams")

    # ---- ATO / SIM Swap ----
    if "NEW" in x.get("Device_ID", ""):
        reasons.append("Transaction initiated from a new or unrecognized device")

    if x.get("Hour_of_Day", 12) in [1, 2, 3, 4]:
        reasons.append("Transaction occurred during unusual late-night hours")

    # ---- Amount anomalies ----
    if x.get("Amount_Change_Ratio", 1) > 5:
        reasons.append("Transaction amount is significantly higher than usual")

    # ---- Trust signals ----
    if x.get("Is_First_Time_Receiver", 0) == 1:
        reasons.append("First-time payment to this receiver")

    if any(w in x.get("Receiver_ID", "").lower() for w in ["support", "kyc", "care"]):
        reasons.append("Receiver ID resembles a known impersonation pattern")

    # ---- Velocity & geo ----
    if x.get("Txn_Count_1h", 0) >= 5:
        reasons.append("Multiple transactions attempted in a short period")

    if x.get("Geo_Jump", 0) > 500:
        reasons.append("Unusual geographic location change detected")

    return reasons

# MAIN EXPLANATION FUNCTION
def explain_single_transaction(raw_input_dict):
    vectorizer = get_vectorizer()
    scaler = get_scaler()
    pass_agg_clf = get_model()
    x = raw_input_dict.copy()

    # Derived features
    x["VPA_Keyword_Match"] = int(
        any(w in x.get("Receiver_ID", "").lower() for w in ["support", "care", "kyc"])
    )
    x["Is_New_Device"] = int("NEW" in x.get("Device_ID", ""))
    x["Amount_Change_Ratio"] = float(x["Amount"] / x["Avg_Transaction_Value"])

    # Prediction
    x_vec = vectorizer.transform([x])
    X_scaled = scaler.transform(x_vec)
    y_pred = pass_agg_clf.predict(X_scaled)[0]

    # Output
    if y_pred == "Legit":
        reason_text = "No significant fraud indicators detected."
    else:
        reasons = generate_reasons(x)
        reason_text = "\n".join([f"• {r}" for r in reasons[:3]])

    return y_pred, f"{reason_text}"
