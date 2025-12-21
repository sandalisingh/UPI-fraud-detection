from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from backend.fraud_simulation.Explanation import explain_single_transaction
from fastapi.exceptions import RequestValidationError

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
    Amount: float = Field(gt=0, le=100000)
    Transaction_Type: str
    Channel: str
    Sender_ID: str
    Receiver_ID: str
    Device_ID: str
    Geo_Jump: int = Field(ge=0, le=5000)
    Network_Type: str
    Is_First_Time_Receiver: int = Field(ge=0, le=1)
    Sender_Account_Age: int = Field(ge=0)
    Avg_Transaction_Value: float = Field(ge=0)
    Txn_Count_1h: int = Field(ge=0)
    Time_Since_Last_Txn: int = Field(ge=0)

    @validator("Timestamp")
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", ""))
            return v
        except Exception:
            raise ValueError("Timestamp must be ISO-8601 format")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    messages = []

    for err in exc.errors():
        field = err["loc"][-1]
        error_type = err["type"]
        ctx = err.get("ctx", {})

        # Custom messages per rule
        if error_type == "less_than_equal":
            messages.append(f"{field} should be less than {ctx.get('le')}")
        elif error_type == "greater_than":
            messages.append(f"{field} should be greater than {ctx.get('gt')}")
        elif error_type == "value_error.missing":
            messages.append(f"{field} is required")
        else:
            messages.append(f"Invalid value for {field}")

    return JSONResponse(
        status_code=422,
        content={
            "error": messages[0] if messages else "Invalid request"
        }
    )

@app.exception_handler(RuntimeError)
async def runtime_exception_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )

# Prediction endpoint
@app.post("/predict")
def predict_fraud_V1(txn: Transaction):
    try:
        result = explain_single_transaction(txn.dict())
        return result

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(ve)}"
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not found on server"
        )

    except Exception as e:
        # Catch-all safeguard
        raise HTTPException(
            status_code=500,
            detail="Internal fraud evaluation error"
        )

