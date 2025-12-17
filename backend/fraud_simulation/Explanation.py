import shap
import numpy as np
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

def vpa_semantic_risk(vpa):
    BRAND_KEYWORDS = [
        "refund", "cashback", "reward", "support",
        "kyc", "care", "help", "alerts"
    ]
    return int(any(k in vpa.lower() for k in BRAND_KEYWORDS))

def softmax(logits):
    logits = logits - np.max(logits)  # numerical stability
    exp = np.exp(logits)
    return exp / np.sum(exp)

# MAIN EXPLANATION FUNCTION
def explain_single_transaction(raw_input_dict):
    vectorizer = get_vectorizer()
    scaler = get_scaler()
    model = get_model()
    shap_bg = np.load("backend/fraud_simulation/shap_background.npy")

    x = raw_input_dict.copy()

    # ---- Derived features ----
    x["VPA_Keyword_Match"] = vpa_semantic_risk(x.get("Receiver_ID", ""))
    x["Is_New_Device"] = int("NEW" in x.get("Device_ID", ""))
    x["Amount_Change_Ratio"] = float(x["Amount"] / x["Avg_Transaction_Value"])

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
    filtered = [
        (f, v, X_scaled[0, i])
        for i, (f, v) in enumerate(zip(feature_names, shap_vals))
        if abs(v) >= 0.005
    ]

    # Sort by impact
    filtered.sort(key=lambda x: abs(x[1]), reverse=True)

    top_shap_reasons = [
        f"{f} increased likelihood of {y_pred}"
        if v > 0 else
        f"{f} reduced likelihood of {y_pred}"
        for f, v, _ in filtered[:5]
    ]

    # ---- Rule reasons ----
    rule_reasons = (
        ["No significant fraud indicators detected."]
        if y_pred == "Legit"
        else generate_reasons(x)[:3]
    )

    return {
        "fraud_type": y_pred,
        "risk_percent": risk_pct,
        "shap_reasons": top_shap_reasons,
        "explanation": rule_reasons
    }
