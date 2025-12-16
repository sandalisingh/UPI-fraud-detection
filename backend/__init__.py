import os
import json
import joblib
from huggingface_hub import hf_hub_download

# Hugging Face config
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "sandalisingh/upi-fraud-models"

_model_V1 = None
_model_V2 = None
_scaler = None
_feature_names = None

def get_HGDB_model():
    global _model_V1
    if _model_V1 is None:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="fraud_model_histGDB.pkl",
            token=HF_TOKEN
        )
        _model_V1 = joblib.load(model_path)
    return _model_V1

def get_Hoeffding_tree_model():
    global _model_V2
    if _model_V2 is None:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="fraud_detection_model_hoeffdingtree.pkl",
            token=HF_TOKEN
        )
        _model_V2 = joblib.load(model_path)
    return _model_V2

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
