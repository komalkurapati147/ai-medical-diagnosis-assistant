# AI Medical Diagnosis Assistant

An end-to-end ML system that predicts a likely disease from reported symptoms,
explains the prediction with SHAP, generates a plain-English narrative with an
LLM (Gemini or OpenAI), stores every report in PostgreSQL, and produces a
downloadable PDF report — all exposed through a FastAPI service with Swagger UI.

> ⚠️ Educational project. Not a certified medical device. Do not use for real
> clinical decisions.

## Architecture

```
Client (Swagger UI / curl / frontend)
        │  POST /api/v1/predict {symptoms}
        ▼
FastAPI (app/main.py, app/routers/)
        │
        ▼
Preprocessing (app/ml/preprocess.py) -> aligns input to trained feature schema
        ▼
XGBoost Classifier (app/ml/predict.py) -> predicted disease + confidence
        ▼
SHAP TreeExplainer (app/ml/explainability.py) -> top contributing symptoms
        ▼
LLM Service (app/services/llm_service.py, Gemini/OpenAI) -> plain-English summary
        ▼
PDF Service (app/services/pdf_service.py, reportlab) -> downloadable report
        ▼
PostgreSQL (app/db/) -> persists every report
```

## Tech stack

| Layer              | Technology                          |
|--------------------|--------------------------------------|
| API framework      | FastAPI + Uvicorn                    |
| ML model           | XGBoost (multi-class classification) |
| Explainability     | SHAP (TreeExplainer)                 |
| LLM narrative      | Google Gemini / OpenAI GPT           |
| Report generation  | ReportLab (PDF)                      |
| Database           | PostgreSQL + SQLAlchemy ORM          |
| Containerization   | Docker + docker-compose              |
| Deployment         | Render.com (render.yaml)             |

## Project structure

```
ai-medical-diagnosis-assistant/
├── app/
│   ├── main.py                 # FastAPI app, router registration, health check
│   ├── config.py               # env-driven settings (pydantic-settings)
│   ├── schemas.py              # Pydantic request/response models
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine/session
│   │   ├── models.py           # DiagnosisReport ORM model
│   │   └── crud.py             # DB read/write helpers
│   ├── ml/
│   │   ├── train_model.py      # dataset synth + XGBoost training script
│   │   ├── preprocess.py       # raw symptoms -> model feature vector
│   │   ├── predict.py          # loads model, runs inference
│   │   └── explainability.py   # SHAP TreeExplainer wrapper
│   ├── services/
│   │   ├── llm_service.py      # Gemini/OpenAI abstraction
│   │   └── pdf_service.py      # PDF report builder
│   └── routers/
│       ├── predict.py          # POST /api/v1/predict
│       └── reports.py          # GET  /api/v1/reports, /{id}, /{id}/pdf
├── data/                        # generated training dataset (csv)
├── saved_models/                 # trained model + encoder + feature list (.pkl)
├── generated_reports/            # generated PDF reports
├── tests/test_predict.py
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
└── .env.example
```

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # fill in GEMINI_API_KEY or OPENAI_API_KEY (optional)

# 1. Train the model (creates saved_models/*.pkl + data/*.csv)
python -m app.ml.train_model

# 2. Start Postgres (or point DATABASE_URL at your own instance)
docker run -d --name pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=medical_diagnosis -p 5432:5432 postgres:16-alpine

# 3. Run the API
uvicorn app.main:app --reload
# Swagger UI -> http://localhost:8000/docs
```

## Running with Docker Compose (recommended)

```bash
cp .env.example .env
docker-compose up --build
# API -> http://localhost:8000/docs
```

## API endpoints

| Method | Path                          | Description                              |
|--------|--------------------------------|-------------------------------------------|
| POST   | /api/v1/predict                | Predict disease from symptoms, returns PDF link |
| GET    | /api/v1/reports                | List past reports                         |
| GET    | /api/v1/reports/{id}           | Get a single report's full detail         |
| GET    | /api/v1/reports/{id}/pdf       | Download that report's PDF                |
| GET    | /health                        | Liveness check                            |

### Example request

```json
POST /api/v1/predict
{
  "patient_name": "John Doe",
  "age": 34,
  "gender": "male",
  "symptoms": {
    "fever": 1, "cough": 1, "fatigue": 1, "shortness_of_breath": 1,
    "loss_of_smell": 1, "sore_throat": 0
  }
}
```

## Deployment (Render)

`render.yaml` defines a Docker web service + a managed PostgreSQL instance.
Push to GitHub, connect the repo on Render as a Blueprint, set
`GEMINI_API_KEY`/`OPENAI_API_KEY` as secrets, and Render builds the Dockerfile
(which also trains the model at build time) and deploys automatically.

## Notes on the dataset

`app/ml/train_model.py` synthesizes a symptom-disease dataset from
hand-authored symptom probability profiles for 8 conditions so the whole
project is runnable offline with no external downloads. Swap in a real
dataset (e.g. Kaggle's "Disease Prediction Using Machine Learning") by
replacing `synthesize_dataset()` with a `pd.read_csv(...)` call — the rest of
the pipeline (train/test split, XGBoost, SHAP, API) is unchanged.
