import os
import json
import joblib
from huggingface_hub import hf_hub_download

# Hugging Face config
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "sandalisingh/upi-fraud-models"

_model = None
_scaler = None
_feature_names = None

def get_HGDB_model():
    global _model
    if _model is None:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="fraud_model_histGDB.pkl",
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


def get_feature_names():
    global _feature_names
    if _feature_names is None:
        feature_names_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="feature_names.json",
            token=HF_TOKEN
        )
        with open(feature_names_path, "r") as f:
            _feature_names = json.load(f)
    return _feature_names
