from backend.fraud_simulation.Explanation import explain_user_transaction_hoeffding
from backend.fraud_simulation.Explanation import model

# TEST CASE
sample_txn = {
    "Sender_ID": "U001",
    "Receiver_ID": "mule.account@upi",
    "Amount": 45000,
    "Transaction_Type": "P2P",
    "Channel": "Manual_VPA",
    "Device_ID": "D_NEW_99",
    "Geo_Jump": 1200,
    "Amount_Change_Ratio": 15.5,
    "Txn_Count_1h": 8,
    "Hour_of_Day": 3,
    "Network_Type": "Public_WiFi",
    "Is_First_Time_Receiver": 1,
    "Sender_Account_Age": 5,
    "Avg_Transaction_Value": 2900,
    "Time_Since_Last_Txn": 10
}

print(explain_user_transaction_hoeffding(model, sample_txn))
