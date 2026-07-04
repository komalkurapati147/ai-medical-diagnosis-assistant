FROM python:3.11-slim

WORKDIR /code

# System deps needed by psycopg2 / xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at image build time so saved_models/ is populated
# (skip this and mount a pre-trained model in real production instead).
RUN python -m app.ml.train_model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
