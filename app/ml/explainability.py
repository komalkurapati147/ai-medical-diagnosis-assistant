"""
SHAP explainability for the XGBoost model.

`TreeExplainer` is used because it's the fast, exact SHAP algorithm for
tree-based models (as opposed to KernelExplainer, which is a much
slower model-agnostic approximation). The explainer is built once and
reused, since building it re-reads the trained trees.
"""
from typing import List, Dict
import numpy as np
import pandas as pd
import shap

from app.ml.predict import _lazy_load

_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        model, _, _ = _lazy_load()
        _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_prediction(feature_df: pd.DataFrame, predicted_class_idx: int, top_k: int = 5) -> List[Dict]:
    """
    Returns the top_k features (symptoms) that most influenced the
    prediction for the specific predicted class, sorted by absolute
    SHAP value (impact).
    """
    explainer = _get_explainer()
    shap_values = explainer.shap_values(feature_df)

    # xgboost multi-class TreeExplainer returns a list[array] (one per class)
    # or a single 3D array depending on SHAP version; handle both.
    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class_idx][0]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:
            class_shap = arr[0, :, predicted_class_idx]
        else:
            class_shap = arr[0]

    contributions = []
    for feature, value, impact in zip(feature_df.columns, feature_df.iloc[0].values, class_shap):
        contributions.append({
            "feature": feature,
            "value": float(value),
            "impact": float(impact),
        })

    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contributions[:top_k]
