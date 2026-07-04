"""
ORM models. A single `DiagnosisReport` table stores every prediction
made by the system, along with the SHAP explanation, the LLM narrative,
and a pointer to the generated PDF file.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, JSON, Text


from app.db.database import Base


class DiagnosisReport(Base):
    __tablename__ = "diagnosis_reports"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name = Column(String, nullable=True)
    symptoms = Column(JSON, nullable=False)          # raw symptom payload from the request
    predicted_disease = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    shap_explanation = Column(JSON, nullable=True)    # top contributing features
    llm_summary = Column(Text, nullable=True)         # human-readable narrative from Gemini/OpenAI
    pdf_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
