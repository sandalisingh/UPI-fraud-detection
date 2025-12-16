#%%
import random
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
from river import tree, preprocessing, metrics, compose
import joblib

#%%# CONFIGURATION 
MODEL_PATH = "fraud_detection_model_hoeffdingtree.pkl"
N_USERS = 1000
N_SIMULATION_HOURS = 24
BASE_FRAUD_RATE = 0.01

START_TIME = datetime(2025, 1, 1, 0, 0, 0)

TRANSACTION_TYPES = ["P2P", "P2M", "Bill_Pay", "Collect_Request"]
CHANNELS = ["QR_Scan", "Intent_Link", "Manual_VPA"]
NETWORK_TYPES = ["4G", "5G", "Public_WiFi"]
FRAUD_TYPES = ["None","Phishing", "QR_Scam", "Collect_Request_Scam", "Identity_Theft"]

#%%# FRAUD EVOLUTION PHASES (in minutes)
FRAUD_PHASES = {
    0: {  # Phase 0: Baseline
        "active_frauds": ["Phishing", "QR_Scam"],
        "amount_multiplier": (5, 15),
        "geo_jump_range": (100, 2000)
    },
    3000: {  # Phase 1: Collect abuse emerges
        "active_frauds": ["Phishing", "QR_Scam", "Collect_Request_Scam"],
        "amount_multiplier": (8, 20),
        "geo_jump_range": (500, 2500)
    },
    6000: {  # Phase 2: Identity theft + SIM swap
        "active_frauds": [
            "Phishing", "Collect_Request_Scam",
            "Identity_Theft", "SIM_Swap_ATO"
        ],
        "amount_multiplier": (10, 30),
        "geo_jump_range": (800, 4000)
    },
    9000: {  # Phase 3: Adaptive fraud (low & stealthy)
        "active_frauds": [
            "VPA_Mimicry", "SIM_Swap_ATO"
        ],
        "amount_multiplier": (2, 6),  # Mimic legit behavior
        "geo_jump_range": (50, 300)
    }
}

#%%# USER NETWORK INITIALIZATION

def init_users(n):
    users = [f"U{str(i).zfill(5)}" for i in range(n)]
    initial_device = f"D{random.randint(100, 999)}"

    state = {
        u: {
            "account_age": random.randint(1, 2500),
            "avg_txn_value": random.randint(300, 2000),
            "device_id": f"D{random.randint(1, n//3)}",
            "historical_device": initial_device, 
            "last_txn_time": None,
            "last_amounts": deque(maxlen=50),
            "txn_history_1h": deque()
        }
        for u in users
    }
    return users, state

#%%# VELOCITY FEATURES

def txn_count_last_hour(user_state, current_time):
    # Remove transactions older than 1 hour
    q = user_state["txn_history_1h"]
    while q and (current_time - q[0]).seconds > 3600:
        q.popleft()
    return len(q)

def amount_change_ratio(user_state, amount):
    if not user_state["last_amounts"]:
        return 1.0
    
    # Calculate average of last 30 transactions
    avg_30 = np.mean(user_state["last_amounts"])

    return round(amount / max(avg_30, 1), 2)

def time_since_last_txn(user_state, current_time):
    if not user_state["last_txn_time"]:
        return 99999
    return int((current_time - user_state["last_txn_time"]).seconds)

#%%# FRAUD PHASE MANAGEMENT

def get_current_fraud_phase(t):
    phase_keys = sorted(FRAUD_PHASES.keys())
    for i in range(len(phase_keys) - 1, -1, -1):
        if t >= phase_keys[i]:
            return FRAUD_PHASES[phase_keys[i]]
    return FRAUD_PHASES[0]

def dynamic_fraud_probability(t, base_rate):
    # Fraud waves + randomness
    seasonal_boost = 0.5 * np.sin(t / 1500)
    noise = random.uniform(-0.3, 0.3)
    return max(0, base_rate * (1 + seasonal_boost + noise))

#%%# TRANSACTION GENERATOR

def generate_transaction(t, users, state):
    current_time = START_TIME + timedelta(minutes=t)
    sender = random.choice(users)
    sender_state = state[sender]

    phase = get_current_fraud_phase(t)
    fraud_prob = dynamic_fraud_probability(t, BASE_FRAUD_RATE)

    is_fraud = int(random.random() < fraud_prob)
    fraud_type = None

    if is_fraud:
        fraud_type = random.choice(phase["active_frauds"])

    receiver = random.choice([u for u in users if u != sender])
    amount = int(np.random.normal(sender_state["avg_txn_value"], 200))
    channel = "Manual_VPA"
    txn_type = "P2P"
    device_id = sender_state["device_id"]
    geo_jump = random.randint(0, 10)
    is_first_time = 0
    network_type = random.choice(NETWORK_TYPES)

    if is_fraud:
        is_first_time = 1

        amt_low, amt_high = phase["amount_multiplier"]
        geo_low, geo_high = phase["geo_jump_range"]

        # FRAUD MUTATION LOGIC
        if fraud_type == "Phishing":
            channel = random.choice(["Intent_Link", "Manual_VPA"])  # evolving
            amount = int(sender_state["avg_txn_value"] * random.uniform(amt_low, amt_high))
            receiver = "UR_PHISH_MULE"

        elif fraud_type == "QR_Scam":
            channel = "QR_Scan"
            txn_type = random.choice(["P2M", "P2P"])  # evasion
            geo_jump = random.randint(geo_low, geo_high)
            amount = random.randint(800, 12000)

        elif fraud_type == "Collect_Request_Scam":
            txn_type = "Collect_Request"
            amount = random.randint(3000, 25000)

        elif fraud_type == "Identity_Theft":
            device_id = f"D_NEW_{random.randint(100,999)}"
            current_time = current_time.replace(hour=random.randint(1, 4))
            geo_jump = random.randint(geo_low, geo_high)
            amount = int(sender_state["avg_txn_value"] * random.uniform(amt_low, amt_high))

        elif fraud_type == "SIM_Swap_ATO":
            device_id = f"NEW_DEV_{random.randint(1000,9999)}"
            current_time = current_time.replace(hour=random.randint(1, 4))
            geo_jump = random.randint(geo_low, geo_high)
            amount = int(sender_state["avg_txn_value"] * random.uniform(amt_low, amt_high))

        elif fraud_type == "VPA_Mimicry":
            fake_entities = [
                "sbi.support@upi", "axis.kyc@upi",
                "paytm.rewards@upi"
            ]
            receiver = random.choice(fake_entities)
            txn_type = "Collect_Request"
            amount = random.randint(800, 4000)

    # velocity features
    amt_ratio = round(amount / (sender_state["avg_txn_value"] + 1), 2)

    txn = {
        "Transaction_ID": str(uuid.uuid4()),
        "Timestamp": current_time,
        "Amount": amount,
        "Transaction_Type": txn_type,
        "Channel": channel,
        "Sender_ID": sender,
        "Receiver_ID": receiver,
        "Device_ID": device_id,
        "Geo_Jump": geo_jump,
        "Network_Type": network_type,
        "Amount_Change_Ratio": amt_ratio,
        "Is_First_Time_Receiver": is_first_time,
        "Sender_Account_Age": sender_state["account_age"],
        "Avg_Transaction_Value": sender_state["avg_txn_value"],
        "Txn_Count_1h": txn_count_last_hour(sender_state, current_time),
        "Time_Since_Last_Txn": time_since_last_txn(sender_state, current_time) 
    }

    Transaction_Label = fraud_type if is_fraud else "Legit"
    return txn, Transaction_Label

#%%# ONLINE LEARNING MODEL

def init_model():
    numeric_features = [
        "Amount",
        "Sender_Account_Age",
        "Avg_Transaction_Value",
        "Geo_Jump",
        "Txn_Count_1h",
        "Amount_Change_Ratio",
        "Time_Since_Last_Txn",
        "Is_First_Time_Receiver"
    ]

    categorical_features = [
        "Transaction_Type",
        "Channel",
        "Network_Type"
    ]

    # Numeric pipeline
    num_pipe = compose.Select(*numeric_features) | preprocessing.StandardScaler()

    # Categorical pipeline
    cat_pipe = compose.Select(*categorical_features) | preprocessing.OneHotEncoder()

    # Combine numeric + categorical, then attach classifier
    model = (num_pipe + cat_pipe) | tree.HoeffdingTreeClassifier(
        grace_period=200,
        delta=1e-5,
        leaf_prediction="nb"
    )

    metric = metric = metrics.MacroF1()
    return model, metric

#%%# SIMULATION RUN

def run_simulation(n_users, n_simulation_hours, base_fraud_rate):
    global N_SIMULATION_HOURS, N_USERS, BASE_FRAUD_RATE
    N_USERS = n_users
    N_SIMULATION_HOURS = n_simulation_hours 
    BASE_FRAUD_RATE = base_fraud_rate

    users, state = init_users(N_USERS)
    model, metric = init_model()

    records = []
    acc_curve = []

    for t in range(N_SIMULATION_HOURS * 60):
        txn, txn_label = generate_transaction(t, users, state)

        x = {
            "Amount": txn["Amount"],
            "Sender_Account_Age": txn["Sender_Account_Age"],
            "Avg_Transaction_Value": txn["Avg_Transaction_Value"],
            "Geo_Jump": txn["Geo_Jump"],
            "Txn_Count_1h": txn["Txn_Count_1h"],
            "Amount_Change_Ratio": txn["Amount_Change_Ratio"],
            "Time_Since_Last_Txn": txn["Time_Since_Last_Txn"],
            "Is_First_Time_Receiver": txn["Is_First_Time_Receiver"],
            "Transaction_Type": txn["Transaction_Type"],
            "Channel": txn["Channel"],
            "Is_New_Device": int(txn["Device_ID"] != state[txn["Sender_ID"]]["historical_device"]),
            "VPA_Keyword_Match": int(any(word in txn["Receiver_ID"] for word in ['support', 'care', 'kyc'])),
            "Is_First_Time_Receiver": txn["Is_First_Time_Receiver"],
            "Hour_of_Day": txn["Timestamp"].hour,  # SIM swaps often happen at night
            "Network_Type": txn["Network_Type"]
        }

        y_pred = model.predict_one(x)
        if y_pred is not None: 
            metric.update(txn_label, y_pred)
            acc_curve.append(metric.get())

        model.learn_one(x, txn_label)
        records.append(txn)

        if t % 1000 == 0:
            print(f"- Simulated {t} minutes")

    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully")

    return pd.DataFrame(records), acc_curve