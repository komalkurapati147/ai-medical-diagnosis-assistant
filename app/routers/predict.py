"""
Core prediction endpoint. Orchestrates:
  1. preprocessing        -> app/ml/preprocess.py
  2. XGBoost inference     -> app/ml/predict.py
  3. SHAP explainability   -> app/ml/explainability.py
  4. LLM narrative         -> app/services/llm_service.py
  5. PDF generation        -> app/services/pdf_service.py
  6. Persist to PostgreSQL -> app/db/crud.py
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.schemas import SymptomInput, PredictionResponse, ShapContribution
from app.db.database import get_db
from app.db import crud
from app.ml.preprocess import build_feature_vector
from app.ml.predict import predict as run_prediction, get_feature_columns
from app.ml.explainability import explain_prediction
from app.services.llm_service import generate_explanation
from app.services.pdf_service import generate_pdf_report

router = APIRouter(prefix="/api/v1", tags=["Diagnosis"])


@router.post("/predict", response_model=PredictionResponse)
def predict_disease(payload: SymptomInput, db: Session = Depends(get_db)):
    if not payload.symptoms or all(v == 0 for v in payload.symptoms.values()):
        raise HTTPException(status_code=400, detail="At least one symptom must be present (value=1).")

    feature_columns = get_feature_columns()
    feature_df = build_feature_vector(payload.symptoms, feature_columns)

    predicted_disease, confidence, all_probs = run_prediction(feature_df)

    # Need the class index (not just the label) for SHAP's per-class output
    from app.ml.predict import _lazy_load
    _, encoder, _ = _lazy_load()
    predicted_idx = int(list(encoder.classes_).index(predicted_disease))

    top_symptoms = explain_prediction(feature_df, predicted_idx, top_k=5)

    llm_summary = generate_explanation(predicted_disease, confidence, top_symptoms)

    report_id = str(uuid.uuid4())
    pdf_path = generate_pdf_report(
        report_id=report_id,
        patient_name=payload.patient_name,
        symptoms=payload.symptoms,
        predicted_disease=predicted_disease,
        confidence_score=confidence,
        top_symptoms=top_symptoms,
        llm_summary=llm_summary,
    )

    saved = crud.create_report(
        db,
        patient_name=payload.patient_name,
        symptoms=payload.symptoms,
        predicted_disease=predicted_disease,
        confidence_score=confidence,
        shap_explanation=top_symptoms,
        llm_summary=llm_summary,
        pdf_path=pdf_path,
    )

    return PredictionResponse(
        report_id=str(saved.id),
        predicted_disease=predicted_disease,
        confidence_score=confidence,
        top_symptoms=[ShapContribution(**s) for s in top_symptoms],
        llm_summary=llm_summary,
        pdf_url=f"/api/v1/reports/{saved.id}/pdf",
    )
