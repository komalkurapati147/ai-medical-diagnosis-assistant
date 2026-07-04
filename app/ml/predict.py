"""
Loads the trained XGBoost model + label encoder + feature list ONCE at
process startup (module-level singletons) and exposes a `predict()`
function used by the FastAPI router. Loading once avoids the cost of
deserializing the model on every request.
"""
import joblib
import numpy as np
import pandas as pd

from app.config import settings

_model = None
_encoder = None
_feature_columns = None


def _lazy_load():
    global _model, _encoder, _feature_columns
    if _model is None:
        _model = joblib.load(settings.model_path)
        _encoder = joblib.load(settings.encoder_path)
        _feature_columns = joblib.load(settings.features_path)
    return _model, _encoder, _feature_columns


def get_feature_columns():
    _, _, cols = _lazy_load()
    return cols


def predict(feature_df: pd.DataFrame):
    """
    Returns (predicted_disease: str, confidence: float, probabilities: dict)
    """
    model, encoder, _ = _lazy_load()
    proba = model.predict_proba(feature_df)[0]
    pred_idx = int(np.argmax(proba))
    predicted_disease = encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    all_probs = {
        cls: float(p) for cls, p in zip(encoder.classes_, proba)
    }
    return predicted_disease, confidence, all_probs
