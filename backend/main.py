from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.fraud_simulation.Explanation import explain_single_transaction

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
class Transaction(BaseModel):
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
    Is_First_Time_Receiver: int
    Sender_Account_Age: int
    Avg_Transaction_Value: float
    Txn_Count_1h: int
    Time_Since_Last_Txn: int

@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict_fraud_V1(txn: Transaction):
    pred, explanation = explain_single_transaction(txn.dict())
    return {
        "fraud_type": pred,
        "explanation": explanation
    }

