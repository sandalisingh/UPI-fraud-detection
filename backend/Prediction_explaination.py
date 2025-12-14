# %%
from Data_preparation import data_preparation, feature_engineering
import joblib
import json
import shap
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import os

HF_TOKEN = os.getenv("HF_TOKEN")

# Download files from Hugging Face
model_path = hf_hub_download(
    repo_id="sandalisingh/upi-fraud-models",
    filename="fraud_model_histGDB.pkl",
    token=HF_TOKEN
)
scaler_path = hf_hub_download(
    repo_id="sandalisingh/upi-fraud-models",
    filename="scaler.pkl",
    token=HF_TOKEN
)
feature_names_path = hf_hub_download(
    repo_id="sandalisingh/upi-fraud-models",
    filename="feature_names.json",
    token=HF_TOKEN
)

# Load models
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
with open(feature_names_path, "r") as f:
    feature_names = json.load(f)

# %%
feature_descriptions = {
    # Core Transaction Features
    "amount": "Transaction amount is unusual",
    "oldbalanceOrg": "Sender’s account balance before the transaction",
    "oldbalanceDest": "Receiver’s account balance before the transaction",

    # Transaction Type Indicators
    "type_TRANSFER": "Transaction is a TRANSFER, commonly associated with fraud",
    "type_CASH_OUT": "Transaction is a CASH_OUT, often used to withdraw stolen funds",
    "type_DEBIT": "Transaction is a DEBIT payment",
    "type_PAYMENT": "Transaction is a PAYMENT to a merchant or service",

    "high_risk_type": "Transaction type is historically high-risk (TRANSFER or CASH_OUT)",

    # Frequency & Behavioral Patterns
    "txn_freq_user": "Sender has made an unusually high number of transactions",
    "avg_amount_user": "Sender’s typical transaction amount",
    "txn_freq_dest": "Receiver has received transactions from many sources",
    "avg_amount_dest": "Receiver’s typical incoming transaction amount",

    "unique_receivers": "Sender has transferred money to many different receivers",
    "unique_senders": "Receiver has received money from many different senders",

    # Network / Graph-Based Risk Signals
    "dest_sender_degree": "Receiver frequently sends money onward, indicating mule behavior",
    "dest_receiver_degree": "Receiver frequently receives funds from multiple accounts",
    "sender_to_receiver_ratio": "Sender initiates many transfers compared to received ones",

    # Balance Consistency Checks
    "amount_ratio": "Transaction amount is large relative to sender’s available balance",

    "sender_balance_change": "Sender’s balance dropped sharply after the transaction",
    "receiver_balance_change": "Receiver’s balance increased sharply after the transaction",

    "org_balance_mismatch": "Sender’s post-transaction balance does not match expected accounting rules",
    "dest_balance_mismatch": "Receiver’s post-transaction balance does not match expected accounting rules",

    # Time and Velocity-Based Risk
    "is_night": "Transaction occurred during unusual hours (late night / early morning)",

    # Behavioral Red Flags
    "orig_balance_zero": "Sender account had zero balance before the transaction",
    "dest_balance_zero": "Receiver account had zero balance before receiving funds",
    "balance_drained": "Transaction drained most or all of the sender’s account balance",

    # Amount-Based Rule Thresholds
    "amt_less_than_10000": "Transaction amount is small (below ₹10,000)",
    "amt_greater_than_10000": "Transaction amount is between ₹10,000 and ₹50,000",
    "amt_greater_than_50000": "Transaction amount is between ₹50,000 and ₹100,000",
    "amt_greater_than_100000": "Transaction amount is very large (above ₹100,000)",
    "amt_greater_than_500000": "Transaction amount is extremely large (above ₹500,000)",
}


# %%
def shap_explain(model, X_background, X_explain):
    print("\n\n=> SHAP EXPLAINATION")
    print("-------------------------------\n\n")

    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional"
        )

    shap_values = explainer(X_explain, check_additivity=False)

    shap.summary_plot(shap_values, X_explain)

# %%
def generate_reason_text(shap_vals, feature_names, X_single, top_k=3):
    abs_vals = np.abs(shap_vals)
    top_idx = np.argsort(abs_vals)[::-1][:top_k]

    reasons = []
    for idx in top_idx:
        fname = feature_names[idx]
        fvalue = X_single.iloc[0][fname]

        if fname in feature_descriptions:
            desc = feature_descriptions[fname]
        else:
            desc = f"unusual value of feature '{fname}'"

        reasons.append(f"{desc}")

    return reasons

# %%
def explain_user_transaction(raw_input_dict, model=model):
    # convert user input → dataframe
    single_row_df = pd.DataFrame([raw_input_dict])

    # data preprocessing
    single_row_df = feature_engineering(single_row_df)
    single_row_df, feature_names, scaler = data_preparation(single_row_df, useSMOTE=False, want_train_test_split=False)
    single_row_df = pd.DataFrame(scaler.transform(single_row_df), columns=feature_names)
    print(single_row_df)

    # Generate explanation
    prob = model.predict_proba(single_row_df)[0][1]
    pred = model.predict(single_row_df)[0]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(single_row_df)
    shap_vals = shap_values.values[0]
    shap_values.base_values[0]

    reasons = generate_reason_text(shap_vals, feature_names, single_row_df, top_k=3)

    risk_percent = int(prob * 100)

    explanation = ""
    if pred == 1:
        explanation += (
            f"This transaction was flagged as suspicious ({risk_percent}% risk)."
            + "\n\nKey contributing factors:\n"
            + "\n".join([f"► {r}" for r in reasons])
        )
    else:
        explanation += f"This transaction appears legitimate ({risk_percent}% risk)."

    return int(pred), str(explanation)
