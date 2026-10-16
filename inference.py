"""
inference.py
-------------
Loads the trained 1D-CNN + scaler once at app startup, and exposes a
predict_eeg() function that takes a raw 178-value EEG row (list/array)
and returns (label, probability).

Kept separate from predict.py (routes) so the model-loading logic is easy
to test or swap out independently of Flask.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "seizure_cnn.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeizureCNN(nn.Module):
    """Must match the architecture in model/train_cnn.py exactly,
    since we are loading weights trained with that definition."""

    def __init__(self, input_len=178):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(0.4)
        flat_len = input_len // 8
        self.fc1 = nn.Linear(64 * flat_len, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)


_model = None
_scaler = None
_model_load_error = None


def _load_artifacts():
    """Lazily loads model + scaler once. Sets a module-level error message
    if files are missing, so routes can show a friendly message instead of
    crashing the whole app."""
    global _model, _scaler, _model_load_error

    if _model is not None and _scaler is not None:
        return

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        _model_load_error = (
            "Trained model not found. Please run `python model/train_cnn.py "
            "--csv <path_to_dataset.csv>` first to generate model/seizure_cnn.pt "
            "and model/scaler.pkl."
        )
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        input_len = checkpoint.get("input_len", 178)

        model = SeizureCNN(input_len=input_len).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        _model = model
        _scaler = scaler
        _model_load_error = None
    except Exception as e:
        _model_load_error = f"Failed to load model: {e}"


def get_model_status():
    """Returns None if the model is loaded and ready, otherwise an error string."""
    _load_artifacts()
    return _model_load_error


def predict_eeg(values):
    """
    values: list/array of EEG signal floats (expected length = model's input_len,
            178 for the UCI dataset).

    Returns: dict with keys: label ("Seizure" / "Non-Seizure"), probability (float 0-1)
    Raises: ValueError if model isn't loaded or input shape is wrong.
    """
    _load_artifacts()
    if _model is None or _scaler is None:
        raise ValueError(_model_load_error or "Model is not loaded.")

    values = np.asarray(values, dtype=np.float32).reshape(1, -1)
    expected_len = _scaler.mean_.shape[0]
    if values.shape[1] != expected_len:
        raise ValueError(
            f"Expected {expected_len} EEG values per row, got {values.shape[1]}."
        )

    scaled = _scaler.transform(values)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)  # (1, 1, L)

    with torch.no_grad():
        logit = _model(x)
        prob = torch.sigmoid(logit).item()

    label = "Seizure" if prob >= 0.5 else "Non-Seizure"
    return {"label": label, "probability": prob}
