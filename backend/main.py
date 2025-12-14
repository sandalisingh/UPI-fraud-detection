from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.Prediction_explanation import explain_user_transaction

# FastAPI app
app = FastAPI(title="UPI Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class Transaction(BaseModel):
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

# Prediction endpoint
@app.post("/predict")
def predict_fraud(txn: Transaction):
    pred, explanation = explain_user_transaction(txn.dict())
    return {
        "is_fraud": pred,
        "explanation": explanation
    }
