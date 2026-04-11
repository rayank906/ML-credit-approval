"""Decision strength, risk blending, and human-readable factor explanations."""

from __future__ import annotations

from typing import Any

from features import (
    EDUCATION_TYPES,
    FAMILY_STATUS,
    FEATURE_LABELS,
    FEATURE_ORDER,
    HOUSING_TYPES,
    INCOME_TYPES,
    OCCUPATION_TYPES,
)


def get_decision_strength(prob_unsafe: float) -> dict[str, Any]:
    """Classify the strength of the approval/rejection decision."""
    if prob_unsafe < 0.10:
        return {"label": "Strong Approval", "color": "#16a34a", "emoji": "check",
                "description": "Very low risk — strong candidate for approval."}
    if prob_unsafe < 0.25:
        return {"label": "Likely Approval", "color": "#65a30d", "emoji": "check",
                "description": "Low risk profile. Approval is likely."}
    if prob_unsafe < 0.40:
        return {"label": "Leaning Approval", "color": "#ca8a04", "emoji": "minus",
                "description": "Moderate risk. Could go either way, leaning approve."}
    if prob_unsafe < 0.55:
        return {"label": "Borderline", "color": "#d97706", "emoji": "minus",
                "description": "On the fence — additional review recommended."}
    if prob_unsafe < 0.70:
        return {"label": "Leaning Rejection", "color": "#ea580c", "emoji": "minus",
                "description": "Elevated risk. Leaning toward rejection."}
    if prob_unsafe < 0.85:
        return {"label": "Likely Rejection", "color": "#dc2626", "emoji": "x",
                "description": "High risk profile. Rejection is likely."}
    return {"label": "Strong Rejection", "color": "#991b1b", "emoji": "x",
            "description": "Very high risk — strong candidate for rejection."}


def blend_risk(prob_unsafe: float, credit_score: int, weight_credit_score: float = 0.65) -> float:
    """
    Blend model risk with credit score. Higher credit score -> lower blended risk.
    weight_credit_score is the share of the decision driven by the credit score.
    """
    credit_norm = (credit_score - 300) / 550  # 300-850 -> 0-1
    credit_norm = max(0.0, min(1.0, credit_norm))
    return (weight_credit_score * (1 - credit_norm)) + ((1 - weight_credit_score) * prob_unsafe)


def _human_readable_value(feat_name: str, val: Any) -> str:
    """Convert an encoded value back to a human-readable string."""
    mappings = {
        "CODE_GENDER": {0: "Male", 1: "Female"},
        "FLAG_OWN_CAR": {0: "No", 1: "Yes"},
        "FLAG_OWN_REALTY": {0: "No", 1: "Yes"},
        "FLAG_WORK_PHONE": {0: "No", 1: "Yes"},
        "FLAG_PHONE": {0: "No", 1: "Yes"},
        "FLAG_EMAIL": {0: "No", 1: "Yes"},
        "NAME_INCOME_TYPE": INCOME_TYPES,
        "NAME_EDUCATION_TYPE": EDUCATION_TYPES,
        "NAME_FAMILY_STATUS": FAMILY_STATUS,
        "NAME_HOUSING_TYPE": HOUSING_TYPES,
        "OCCUPATION_TYPE": OCCUPATION_TYPES,
    }
    if feat_name in mappings:
        return str(mappings[feat_name].get(int(val), val))
    if feat_name == "AMT_INCOME_TOTAL":
        return f"${val:,.0f}"
    if feat_name == "AGE_YEARS":
        return f"{val:.1f} years"
    if feat_name == "YEARS_EMPLOYED":
        return f"{val:.1f} years"
    if feat_name == "CREDIT_SCORE":
        return f"{int(val)}"
    return str(val)


def get_top_reasons(pipeline: Any, feature_values: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Use the Random Forest's feature importances + the applicant's values
    to produce human-readable reasons for the decision.
    """
    rf_clf = pipeline.named_steps["clf"]
    importances = rf_clf.feature_importances_
    feature_names = FEATURE_ORDER

    feat_imp = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    reasons: list[dict[str, Any]] = []
    for feat_name, importance in feat_imp[:5]:
        val = feature_values[feat_name]
        label = FEATURE_LABELS.get(feat_name, feat_name)
        human_val = _human_readable_value(feat_name, val)
        reasons.append({
            "feature": label,
            "value": human_val,
            "importance": round(float(importance) * 100, 1),
        })
    return reasons
