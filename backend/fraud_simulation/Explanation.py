import shap
import numpy as np
from backend import get_scaler, get_vectorizer, get_model
from datetime import datetime

# RULE-BASED REASON ENGINE
def generate_reason(f, v):
    # Amount change ratio
    if "Amount_Change_Ratio" in f:
        if v > 5:
            return "Sudden spike in transaction amount relative to historical average. "
        else:
            return "Transaction amount change appears gradual and expected. "
    
    # Amount-based reasoning
    elif "Amount" in f:
        if v > 50000:
            return "High transaction amount observed. "
        elif v > 20000:
            return "Moderately high transaction amount observed. "
        else:
            return "Transaction amount within normal range. "

    # Transaction type
    elif "Transaction_Type" in f:
        if "Collect_Request" in f:
            return "Collect request payments are often misused for refund or cashback scams. "
        elif "Bill_Pay" in f:
            return "Bill payment transactions can be exploited for refund-based scams. "
        elif "P2P" in f:
            return "Direct peer-to-peer transfer, commonly used in impersonation or account takeover frauds. "
        elif "P2M" in f:
            return "Merchant payment channel, sometimes abused in QR or fake merchant scams. "

    # Channel
    elif "Channel" in f:
        if "QR_Scan" in f:
            return "QR-based transaction, a known vector for QR code replacement scams. "
        elif "Intent_Link" in f:
            return "Transaction initiated via intent link, frequently seen in phishing attacks. "
        elif "Manual_VPA" in f:
            return "Manual VPA entry increases risk of sending funds to spoofed or deceptive IDs. "

    # Network type
    elif "Network_Type" in f:
        if "Public_WiFi" in f:
            return "Transaction performed over public Wi-Fi, increasing interception or compromise risk. "
        else:
            return "Transaction executed over a trusted network. "

    # Geo jump
    elif "Geo_Jump" in f:
        if v > 100:
            return "Large geographic location change detected, inconsistent with recent behavior. "
        elif v > 40:
            return "Moderate geo-location shift which may indicate remote access or social engineering. "
        else:
            return "Minimal geographic movement, consistent with normal usage. "

    # First-time receiver
    elif "Is_First_Time_Receiver" in f:
        if v == 1:
            return "Funds sent to a first-time receiver, a common trait in scam and mule accounts. "
        else:
            return "Receiver has prior transaction history with sender. "

    # Sender account age
    elif "Sender_Account_Age" in f:
        if v < 1000:
            return "Relatively new sender account with limited historical trust. "
        else:
            return "Long-standing sender account, typically associated with legitimate behavior. "

    # Avg transaction value
    elif "Avg_Transaction_Value" in f:
        if v > 0:
            return "Deviation from sender’s usual transaction value detected. "
        else:
            return "Transaction aligns with sender’s historical spending pattern. "

    # Txn count last 1h
    elif "Txn_Count_1h" in f:
        if v > 3:
            return "Multiple transactions in a short time window suggest automation or panic-driven activity. "
        else:
            return "Normal transaction frequency observed. "

    # Time since last txn
    elif "Time_Since_Last_Txn" in f:
        if v < 60:
            return "Very short gap between transactions, indicative of scripted or fraudulent behavior. "
        else:
            return "Transaction timing consistent with human usage patterns. "

    # Hour of day
    elif "Hour_of_Day" in f:
        if v < 5 or v > 20:
            return "Transaction executed during off-peak hours, often associated with covert fraud attempts. "
        else:
            return "Transaction occurred during regular active hours. "

    # New device
    elif "Is_New_Device" in f:
        if v == 1:
            return "Transaction initiated from a newly observed device, a strong account takeover signal. "
        else:
            return "Transaction performed from a previously trusted device. "

    # VPA semantic risk
    elif "VPA_Semantic_Risk" in f:
        if v > 0:
            return "Receiver VPA contains impersonation or brand-mimicking patterns. "
        else:
            return "Receiver VPA appears semantically normal. "

    return ""

def vpa_semantic_risk(vpa):
    BRAND_KEYWORDS = [
        "refund", "cashback", "reward", "support",
        "kyc", "care", "help", "alerts"
    ]

    LEGIT_PSP_HANDLES = {
        "ybl", "ibl", "axl", "okhdfcbank", "okicici", "oksbi", "okaxis", "paytm", 
        "ptyes", "ptaxis", "pthdfc", "ptsbi", "upi", "apl", "yapl", "rapl",
        "axisb", "yescred", "icici", "sbi"   
    }

    vpa = vpa.lower()

    # Brand keyword risk
    keyword_risk = any(k in vpa for k in BRAND_KEYWORDS)

    # PSP handle risk
    handle_risk = True
    if "@" in vpa:
        handle = vpa.split("@")[-1]
        handle_risk = handle not in LEGIT_PSP_HANDLES

    # Risk scoring
    return int(keyword_risk) + int(handle_risk) * 10

def softmax(logits):
    logits = logits - np.max(logits)  # numerical stability
    exp = np.exp(logits)
    return exp / np.sum(exp)

def get_current_features(txn):
    txn_time = datetime.fromisoformat(txn["Timestamp"].replace("Z", "+00:00"))
    x = {
        # Transaction info
        "Amount": txn["Amount"],
        "Transaction_Type": txn["Transaction_Type"],
        "Channel": txn["Channel"],
        "Network_Type": txn["Network_Type"],

        # Sender and receiver info
        "Geo_Jump": txn["Geo_Jump"],
        "Is_First_Time_Receiver": txn["Is_First_Time_Receiver"],
        "Sender_Account_Age": txn["Sender_Account_Age"],
        "Avg_Transaction_Value": txn["Avg_Transaction_Value"],
        "Txn_Count_1h": txn["Txn_Count_1h"],
        "Time_Since_Last_Txn": txn["Time_Since_Last_Txn"],

        # Derived features
        "Hour_of_Day": txn_time.hour,
        "Amount_Change_Ratio": round(txn["Amount"] / (txn["Avg_Transaction_Value"] + 1), 2),
        "Is_New_Device": int("NEW" in txn["Device_ID"]),
        "VPA_Semantic_Risk": vpa_semantic_risk(txn["Receiver_ID"]),
    }
    return x

# MAIN EXPLANATION FUNCTION
def explain_single_transaction(raw_input_dict):
    vectorizer = get_vectorizer()
    scaler = get_scaler()
    model = get_model()
    shap_bg = np.load("backend/fraud_simulation/shap_background.npy")

    x = get_current_features(raw_input_dict)

    # ---- Transform ----
    X_vec = vectorizer.transform([x])
    X_scaled = scaler.transform(X_vec)
    feature_names = vectorizer.get_feature_names_out()

    # ---- Predict ----
    y_pred = model.predict(X_scaled)[0]

    # ---- Risk % ----
    logits = model.decision_function(X_scaled)[0]
    probs = softmax(logits)
    prob_map = dict(zip(model.classes_, probs))

    risk_pct = round((1 - prob_map.get("Legit", 0.0)) * 100, 2)

    # ---- SHAP ----
    explainer = shap.LinearExplainer(
        model,
        shap_bg,
        feature_names=feature_names
    )

    shap_values = explainer.shap_values(X_scaled)

    class_idx = list(model.classes_).index(y_pred)

    # Correct extraction
    shap_vals = shap_values[0, :, class_idx]
    base_value = explainer.expected_value[class_idx]

    # ---- Remove zero / near-zero features ----
    filtered = []
    for i, (f, shap_v) in enumerate(zip(feature_names, shap_vals)):
        raw_val = X_vec[0, i]

        # Skip inactive one-hot categorical features
        if "=" in f and raw_val == 0:
            continue

        if abs(shap_v) >= 0.005:
            filtered.append((f, shap_v, raw_val))


    # Sort by impact
    filtered.sort(key=lambda x: abs(x[1]), reverse=True)

    top_shap_reasons = []

    for f, v, _ in filtered[:5]:
        feature_value =  f"{f}={x[f]}" if "=" not in f else f"{f}"
        if v > 0:
            top_shap_reasons.append(generate_reason(f, x[f] if "=" not in f else 0) + f"[{feature_value}]")
        else:
            top_shap_reasons.append(generate_reason(f, x[f] if "=" not in f else 0) + f"[{feature_value}]")

    return {
        "fraud_type": y_pred,
        "risk_percent": risk_pct,
        "reasons": top_shap_reasons,
    }
