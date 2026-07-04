"""
FastAPI application entrypoint.
Run locally with:
    uvicorn app.main:app --reload
Swagger UI is auto-generated at /docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import engine, Base
from app.routers import predict, reports

# Create tables on startup if they don't exist (Alembic migrations are
# the production-grade alternative — see alembic/ folder).
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    description=(
        "An AI-powered clinical decision-support API that predicts a likely "
        "condition from reported symptoms using an XGBoost classifier, explains "
        "the prediction with SHAP, generates a plain-English narrative via an "
        "LLM (Gemini/OpenAI), and produces a downloadable PDF report."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(reports.router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}
