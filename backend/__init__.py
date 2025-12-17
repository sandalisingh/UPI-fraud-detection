import os
import joblib
from huggingface_hub import hf_hub_download

# Hugging Face config
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "sandalisingh/upi-fraud-models"

_model = None
_scaler = None
_vectorizer = None

def get_model():
    global _model
    if _model is None:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="fraud_detection_model.pkl",
            token=HF_TOKEN
        )
        _model = joblib.load(model_path)
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        scaler_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="scaler.pkl",
            token=HF_TOKEN
        )
        _scaler = joblib.load(scaler_path)
    return _scaler


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        vectorizer_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="vectorizer.pkl",
            token=HF_TOKEN
        )
        _vectorizer = joblib.load(vectorizer_path)
    return _vectorizer
