"""
Basic tests. Run with: pytest -q
Assumes the model has already been trained (`python -m app.ml.train_model`).
"""
from app.ml.preprocess import build_feature_vector
from app.ml.predict import predict, get_feature_columns


def test_feature_vector_alignment():
    columns = get_feature_columns()
    raw = {"fever": 1, "unknown_symptom_xyz": 1}
    df = build_feature_vector(raw, columns)
    assert list(df.columns) == columns
    assert df["fever"].iloc[0] == 1
    assert "unknown_symptom_xyz" not in df.columns


def test_prediction_returns_valid_disease():
    columns = get_feature_columns()
    sample = {"fever": 1, "cough": 1, "body_ache": 1, "fatigue": 1}
    df = build_feature_vector(sample, columns)
    disease, confidence, probs = predict(df)
    assert isinstance(disease, str)
    assert 0.0 <= confidence <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-3
