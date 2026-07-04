"""
Centralized application configuration.
Loaded once and reused everywhere via `from app.config import settings`.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "AI Medical Diagnosis Assistant"

    database_url: str = "postgresql://postgres:postgres@localhost:5432/medical_diagnosis"

    llm_provider: str = "gemini"       # "gemini" or "openai"
    gemini_api_key: str = ""
    openai_api_key: str = ""

    model_path: str = "saved_models/xgboost_model.pkl"
    encoder_path: str = "saved_models/label_encoder.pkl"
    features_path: str = "saved_models/feature_columns.pkl"

    reports_dir: str = "generated_reports"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Ensure runtime directories exist
os.makedirs(settings.reports_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.model_path) or ".", exist_ok=True)
