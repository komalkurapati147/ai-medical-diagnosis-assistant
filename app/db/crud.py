"""
Thin data-access layer. Keeping DB writes/reads out of the routers
keeps the API layer testable and the SQL logic in one place.
"""
from sqlalchemy.orm import Session

from app.db.models import DiagnosisReport


def create_report(db: Session, *, patient_name, symptoms, predicted_disease,
                   confidence_score, shap_explanation, llm_summary, pdf_path) -> DiagnosisReport:
    report = DiagnosisReport(
        patient_name=patient_name,
        symptoms=symptoms,
        predicted_disease=predicted_disease,
        confidence_score=confidence_score,
        shap_explanation=shap_explanation,
        llm_summary=llm_summary,
        pdf_path=pdf_path,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_report(db: Session, report_id: str) -> DiagnosisReport | None:
    return db.query(DiagnosisReport).filter(DiagnosisReport.id == report_id).first()


def list_reports(db: Session, limit: int = 50):
    return (
        db.query(DiagnosisReport)
        .order_by(DiagnosisReport.created_at.desc())
        .limit(limit)
        .all()
    )
