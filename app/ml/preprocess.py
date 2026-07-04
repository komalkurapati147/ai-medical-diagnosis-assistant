"""
Converts the raw `{symptom_name: 0/1}` dict from the API request into a
single-row DataFrame whose columns exactly match what the model was
trained on (same order, missing symptoms filled with 0, unknown
symptoms in the input silently ignored so the API never crashes on a
typo from the client).
"""
from typing import Dict, List
import pandas as pd


def build_feature_vector(symptoms: Dict[str, int], feature_columns: List[str]) -> pd.DataFrame:
    row = {col: int(symptoms.get(col, 0)) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)
