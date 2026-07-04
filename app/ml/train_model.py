"""
Training script.

In a real project this reads a curated symptom-disease dataset
(e.g. the Kaggle "Disease Prediction Using Machine Learning" dataset).
To keep this repo self-contained and runnable offline, this script can
also *synthesize* a labeled dataset from a symptom-disease mapping so
`python -m app.ml.train_model` works out of the box with no external
downloads.

Run:
    python -m app.ml.train_model
Produces:
    saved_models/xgboost_model.pkl
    saved_models/label_encoder.pkl
    saved_models/feature_columns.pkl
    data/symptom_disease_dataset.csv
"""
import os
import random

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from app.config import settings

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------
# 1. Disease -> symptom "signature" (probability that symptom appears)
#    This mirrors how public symptom-disease datasets look once pivoted
#    to a wide binary matrix (rows = cases, cols = symptoms).
# ---------------------------------------------------------------------
DISEASE_SYMPTOM_PROFILE = {
    "Common Cold": {
        "cough": 0.8, "sore_throat": 0.8, "runny_nose": 0.9, "fever": 0.3,
        "fatigue": 0.4, "headache": 0.3, "body_ache": 0.3, "shortness_of_breath": 0.05,
        "loss_of_smell": 0.1, "nausea": 0.05, "chest_pain": 0.02, "sneezing": 0.85,
    },
    "Influenza": {
        "cough": 0.7, "sore_throat": 0.5, "runny_nose": 0.4, "fever": 0.9,
        "fatigue": 0.85, "headache": 0.7, "body_ache": 0.85, "shortness_of_breath": 0.2,
        "loss_of_smell": 0.05, "nausea": 0.3, "chest_pain": 0.1, "sneezing": 0.3,
    },
    "COVID-19": {
        "cough": 0.75, "sore_throat": 0.4, "runny_nose": 0.3, "fever": 0.7,
        "fatigue": 0.7, "headache": 0.5, "body_ache": 0.5, "shortness_of_breath": 0.55,
        "loss_of_smell": 0.6, "nausea": 0.2, "chest_pain": 0.25, "sneezing": 0.2,
    },
    "Migraine": {
        "cough": 0.05, "sore_throat": 0.02, "runny_nose": 0.02, "fever": 0.05,
        "fatigue": 0.5, "headache": 0.95, "body_ache": 0.2, "shortness_of_breath": 0.02,
        "loss_of_smell": 0.02, "nausea": 0.6, "chest_pain": 0.05, "sneezing": 0.02,
        "sensitivity_to_light": 0.8, "blurred_vision": 0.4,
    },
    "Gastroenteritis": {
        "cough": 0.05, "sore_throat": 0.05, "runny_nose": 0.02, "fever": 0.4,
        "fatigue": 0.6, "headache": 0.3, "body_ache": 0.3, "shortness_of_breath": 0.02,
        "loss_of_smell": 0.02, "nausea": 0.9, "chest_pain": 0.02, "sneezing": 0.02,
        "diarrhea": 0.9, "abdominal_pain": 0.8, "vomiting": 0.75,
    },
    "Asthma Exacerbation": {
        "cough": 0.8, "sore_throat": 0.1, "runny_nose": 0.1, "fever": 0.1,
        "fatigue": 0.4, "headache": 0.1, "body_ache": 0.1, "shortness_of_breath": 0.9,
        "loss_of_smell": 0.02, "nausea": 0.05, "chest_pain": 0.5, "sneezing": 0.1,
        "wheezing": 0.85,
    },
    "Angina / Cardiac Chest Pain": {
        "cough": 0.1, "sore_throat": 0.02, "runny_nose": 0.02, "fever": 0.05,
        "fatigue": 0.5, "headache": 0.15, "body_ache": 0.1, "shortness_of_breath": 0.6,
        "loss_of_smell": 0.02, "nausea": 0.35, "chest_pain": 0.9, "sneezing": 0.02,
        "sweating": 0.5, "arm_pain": 0.4,
    },
    "Urinary Tract Infection": {
        "cough": 0.05, "sore_throat": 0.02, "runny_nose": 0.02, "fever": 0.4,
        "fatigue": 0.4, "headache": 0.1, "body_ache": 0.15, "shortness_of_breath": 0.02,
        "loss_of_smell": 0.02, "nausea": 0.2, "chest_pain": 0.02, "sneezing": 0.02,
        "burning_urination": 0.9, "frequent_urination": 0.85, "abdominal_pain": 0.4,
    },
}

ALL_SYMPTOMS = sorted({s for profile in DISEASE_SYMPTOM_PROFILE.values() for s in profile})


def synthesize_dataset(n_per_class: int = 300) -> pd.DataFrame:
    rows = []
    for disease, profile in DISEASE_SYMPTOM_PROFILE.items():
        for _ in range(n_per_class):
            row = {}
            for symptom in ALL_SYMPTOMS:
                p = profile.get(symptom, 0.03)  # small noise floor for unrelated symptoms
                row[symptom] = 1 if random.random() < p else 0
            row["disease"] = disease
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def train():
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)

    df = synthesize_dataset()
    df.to_csv("data/symptom_disease_dataset.csv", index=False)
    print(f"Dataset shape: {df.shape}")

    X = df[ALL_SYMPTOMS]
    y_raw = df["disease"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    joblib.dump(model, settings.model_path)
    joblib.dump(encoder, settings.encoder_path)
    joblib.dump(ALL_SYMPTOMS, settings.features_path)
    print(f"Saved model -> {settings.model_path}")
    print(f"Saved encoder -> {settings.encoder_path}")
    print(f"Saved feature columns -> {settings.features_path}")


if __name__ == "__main__":
    train()
