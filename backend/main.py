from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.base_model.Prediction_explanation import explain_user_transaction
from backend.fraud_simulation.Explanation import explain_user_transaction_hoeffding

# FastAPI app
app = FastAPI(title="UPI Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class Transaction_V1(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int

class Transaction_V2(BaseModel):
    Transaction_ID: str
    Timestamp: str
    Amount: float
    Transaction_Type: str
    Channel: str
    Sender_ID: str
    Receiver_ID: str
    Device_ID: str
    Geo_Jump: int
    Network_Type: str
    Amount_Change_Ratio: float
    Is_First_Time_Receiver: int
    Sender_Account_Age: int
    Avg_Transaction_Value: float
    Txn_Count_1h: int
    Time_Since_Last_Txn: int

@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict_V1")
def predict_fraud_V1(txn: Transaction_V1):
    pred, explanation = explain_user_transaction(txn.dict())
    return {
        "is_fraud": pred,
        "explanation": explanation
    }

# Prediction endpoint
@app.post("/predict_V2")
def predict_fraud_V2(txn: Transaction_V2):
    pred, reasons = explain_user_transaction_hoeffding(txn.dict())
    return {
        "fraud_type": pred,
        "reasons": reasons
    }
