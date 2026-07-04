"""
Pydantic models = the request/response contracts shown in Swagger UI.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SymptomInput(BaseModel):
    """
    Incoming request. `symptoms` is a dict of symptom_name -> 0/1
    (present or absent). `patient_name` and `age`/`gender` are optional
    metadata used only for the report header.
    """
    patient_name: Optional[str] = Field(default="Anonymous")
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: Optional[str] = Field(default=None)
    symptoms: Dict[str, int] = Field(
        ...,
        description="Map of symptom name -> 1 (present) / 0 (absent)",
        examples=[{
            "fever": 1, "cough": 1, "fatigue": 1, "headache": 0,
            "shortness_of_breath": 1, "sore_throat": 0, "body_ache": 1,
            "loss_of_smell": 0, "nausea": 0, "chest_pain": 0
        }],
    )


class ShapContribution(BaseModel):
    feature: str
    value: float
    impact: float  # positive = pushes toward the predicted class


class PredictionResponse(BaseModel):
    report_id: str
    predicted_disease: str
    confidence_score: float
    top_symptoms: List[ShapContribution]
    llm_summary: str
    pdf_url: str


class ReportSummary(BaseModel):
    id: str
    patient_name: Optional[str]
    predicted_disease: str
    confidence_score: float
    created_at: str
