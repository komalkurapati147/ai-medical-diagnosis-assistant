"""
LLM abstraction layer.

The XGBoost model + SHAP give us a disease label, a confidence score,
and a ranked list of contributing symptoms — but that's not something
a patient can read comfortably. This module turns that structured
output into a plain-English explanation using either Gemini or OpenAI,
selected via LLM_PROVIDER in the environment. Swapping providers never
touches the router or ML code (Strategy pattern).
"""
from typing import List, Dict
from app.config import settings

PROMPT_TEMPLATE = """You are a medical assistant AI helping a patient understand an
automated preliminary diagnosis. Do NOT provide a definitive diagnosis or prescribe
treatment; always recommend confirming with a licensed physician.

Predicted condition: {disease}
Model confidence: {confidence:.1%}
Top contributing symptoms (from SHAP explainability): {symptoms}

Write a short (4-6 sentence), empathetic, plain-English explanation of:
1. What this predicted condition generally is
2. Why these symptoms likely drove this prediction
3. A clear reminder that this is an AI-assisted preliminary screening, not a
   final diagnosis, and the patient should consult a doctor.
"""


def _format_symptoms(top_symptoms: List[Dict]) -> str:
    present = [s["feature"].replace("_", " ") for s in top_symptoms if s["value"] == 1]
    return ", ".join(present) if present else "a combination of reported symptoms"


def _build_prompt(disease: str, confidence: float, top_symptoms: List[Dict]) -> str:
    return PROMPT_TEMPLATE.format(
        disease=disease,
        confidence=confidence,
        symptoms=_format_symptoms(top_symptoms),
    )


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _fallback_summary(disease: str, confidence: float, top_symptoms: List[Dict]) -> str:
    """Used if no API key is configured, so the API still works end-to-end in a demo/offline setting."""
    symptoms_str = _format_symptoms(top_symptoms)
    return (
        f"Based on the reported symptoms ({symptoms_str}), the model predicts "
        f"'{disease}' with {confidence:.1%} confidence. This is an automated, "
        f"preliminary screening only — it is not a medical diagnosis. Please "
        f"consult a licensed physician for confirmation and treatment."
    )


def generate_explanation(disease: str, confidence: float, top_symptoms: List[Dict]) -> str:
    prompt = _build_prompt(disease, confidence, top_symptoms)
    try:
        if settings.llm_provider == "gemini" and settings.gemini_api_key:
            return _call_gemini(prompt)
        if settings.llm_provider == "openai" and settings.openai_api_key:
            return _call_openai(prompt)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never break the API
        print(f"[llm_service] LLM call failed, using fallback summary: {exc}")

    return _fallback_summary(disease, confidence, top_symptoms)
