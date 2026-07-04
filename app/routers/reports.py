"""
Endpoints to list past reports, fetch one by id, and download its PDF.
Everything reads from PostgreSQL via app/db/crud.py.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import crud
from app.schemas import ReportSummary

router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])


@router.get("", response_model=list[ReportSummary])
def list_reports(limit: int = 50, db: Session = Depends(get_db)):
    reports = crud.list_reports(db, limit=limit)
    return [
        ReportSummary(
            id=str(r.id),
            patient_name=r.patient_name,
            predicted_disease=r.predicted_disease,
            confidence_score=r.confidence_score,
            created_at=r.created_at.isoformat(),
        )
        for r in reports
    ]


@router.get("/{report_id}")
def get_report(report_id: str, db: Session = Depends(get_db)):
    report = crud.get_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "id": str(report.id),
        "patient_name": report.patient_name,
        "symptoms": report.symptoms,
        "predicted_disease": report.predicted_disease,
        "confidence_score": report.confidence_score,
        "shap_explanation": report.shap_explanation,
        "llm_summary": report.llm_summary,
        "created_at": report.created_at.isoformat(),
    }


@router.get("/{report_id}/pdf")
def download_pdf(report_id: str, db: Session = Depends(get_db)):
    report = crud.get_report(db, report_id)
    if not report or not report.pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found for this report")
    return FileResponse(report.pdf_path, media_type="application/pdf",
                         filename=f"diagnosis_report_{report_id}.pdf")
