"""Shared prediction pipeline — used by the HTTP `/predict` endpoint and the Redis worker."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from credit_score import (
    _score_age,
    _score_children,
    _score_education,
    _score_employment,
    _score_family_members,
    _score_housing,
    _score_income,
    _score_income_type,
    _score_occupation,
    compute_credit_score,
)
from decision import blend_risk, get_decision_strength, get_top_reasons
from features import FEATURE_ORDER

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _REPO_ROOT / "model" / "credit_approval_model.joblib"
model = joblib.load(_MODEL_PATH)


def _compute_credit_breakdown(features: dict) -> list[dict]:
    items = [
        ("Income", _score_income(float(features.get("AMT_INCOME_TOTAL", 0)))),
        ("Age", _score_age(float(features.get("AGE_YEARS", 0)))),
        ("Employment Stability", _score_employment(float(features.get("YEARS_EMPLOYED", 0)))),
        ("Property Ownership", 30 if int(features.get("FLAG_OWN_REALTY", 0)) == 1 else 0),
        ("Car Ownership", 10 if int(features.get("FLAG_OWN_CAR", 0)) == 1 else 0),
        ("Children", _score_children(int(features.get("CNT_CHILDREN", 0)))),
        ("Family Size", _score_family_members(int(features.get("CNT_FAM_MEMBERS", 0)))),
        ("Education", _score_education(int(features.get("NAME_EDUCATION_TYPE", 1)))),
        ("Income Type", _score_income_type(int(features.get("NAME_INCOME_TYPE", 0)))),
        ("Housing", _score_housing(int(features.get("NAME_HOUSING_TYPE", 0)))),
        ("Occupation", _score_occupation(int(features.get("OCCUPATION_TYPE", 18)))),
    ]
    return [
        {"factor": name, "points": pts, "direction": "positive" if pts > 0 else ("negative" if pts < 0 else "neutral")}
        for name, pts in items
    ]


def _compute_confidence(prob_unsafe: float) -> float:
    return round(abs(prob_unsafe - 0.5) * 200, 1)


def _generate_risk_tips(features: dict, credit_score: int, prob_risky: float) -> list[str]:
    tips = []
    income = float(features.get("AMT_INCOME_TOTAL", 0))
    years = float(features.get("YEARS_EMPLOYED", 0))
    realty = int(features.get("FLAG_OWN_REALTY", 0))
    car = int(features.get("FLAG_OWN_CAR", 0))
    children = int(features.get("CNT_CHILDREN", 0))
    education = int(features.get("NAME_EDUCATION_TYPE", 1))

    if income < 100000:
        target = 100000
        score_gain = _score_income(target) - _score_income(income)
        if score_gain > 0:
            tips.append(f"Increasing annual income to ${target:,} could add ~{score_gain} points to your credit score.")
    if years < 3:
        score_gain = _score_employment(5) - _score_employment(years)
        if score_gain > 0:
            tips.append(f"Staying at your current job for {3 - years:.0f}+ more years could add ~{score_gain} points.")
    if realty == 0:
        tips.append("Owning property adds +30 points to your credit score — consider this if feasible.")
    if car == 0:
        tips.append("Car ownership adds +10 points to your credit score.")
    if children > 2:
        tips.append("A high number of dependents slightly reduces your score (−20 points).")
    if education < 3:
        score_gain = _score_education(3) - _score_education(education)
        if score_gain > 0:
            tips.append(f"Higher education (degree) could add ~{score_gain} points to your profile.")
    if credit_score < 600:
        tips.append("Your credit score is below average (600). Focus on income stability and reducing obligations.")
    if not tips:
        tips.append("Your profile is already strong. Maintain income stability and employment tenure.")
    return tips


def run_prediction(data: dict) -> dict:
    """Core prediction logic. Accepts a dict with PredictRequest-shaped fields."""
    features = {
        "CODE_GENDER": int(data["gender"]), "FLAG_OWN_CAR": int(data["own_car"]),
        "FLAG_OWN_REALTY": int(data["own_realty"]), "CNT_CHILDREN": int(data["children"]),
        "AMT_INCOME_TOTAL": float(data["income"]), "NAME_INCOME_TYPE": int(data["income_type"]),
        "NAME_EDUCATION_TYPE": int(data["education"]), "NAME_FAMILY_STATUS": int(data["family_status"]),
        "NAME_HOUSING_TYPE": int(data["housing"]), "OCCUPATION_TYPE": int(data["occupation"]),
        "FLAG_WORK_PHONE": int(data["work_phone"]), "FLAG_PHONE": int(data["phone"]),
        "FLAG_EMAIL": int(data["email"]), "CNT_FAM_MEMBERS": int(data["family_members"]),
        "AGE_YEARS": float(data["age"]), "YEARS_EMPLOYED": float(data["years_employed"]),
    }
    credit_score = compute_credit_score(features)
    features["CREDIT_SCORE"] = credit_score
    X = pd.DataFrame([features], columns=FEATURE_ORDER)
    prob_unsafe = float(model.predict_proba(X)[:, 1][0])
    blended = blend_risk(prob_unsafe, credit_score)
    decision = "Rejected" if blended >= 0.4 else "Approved"
    confidence = _compute_confidence(blended)
    strength = get_decision_strength(blended)
    reasons = get_top_reasons(model, features)
    breakdown = _compute_credit_breakdown(features)
    tips = _generate_risk_tips(features, credit_score, blended)
    return {
        "features": features, "credit_score": credit_score,
        "prob_unsafe": prob_unsafe, "blended": blended,
        "decision": decision, "confidence": confidence,
        "strength": strength, "reasons": reasons,
        "breakdown": breakdown, "tips": tips,
    }
