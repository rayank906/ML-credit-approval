"""
Flask web application for Credit Card Approval predictions.
Loads the trained Random Forest model and serves a questionnaire UI.
"""

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = "credit_approval_model.joblib"
model = joblib.load(MODEL_PATH)

# ── Feature mappings (human-readable ↔ encoded) ──────────────────────────

FEATURE_ORDER = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS",
    "AGE_YEARS", "YEARS_EMPLOYED",
]

INCOME_TYPES = {
    0: "Working",
    1: "Commercial Associate",
    2: "Pensioner",
    3: "State Servant",
    4: "Student",
}

EDUCATION_TYPES = {
    0: "Lower Secondary",
    1: "Secondary / Special",
    2: "Incomplete Higher",
    3: "Higher Education",
    4: "Academic Degree",
}

FAMILY_STATUS = {
    0: "Single / Not Married",
    1: "Married",
    2: "Separated",
    3: "Civil Marriage",
    4: "Widow",
}

HOUSING_TYPES = {
    0: "With Parents",
    1: "Rented Apartment",
    2: "Municipal Apartment",
    3: "Co-op Apartment",
    4: "House / Apartment (owned)",
    5: "Office Apartment",
}

OCCUPATION_TYPES = {
    0: "Laborers",
    1: "Core Staff",
    2: "Sales Staff",
    3: "Managers",
    4: "Drivers",
    5: "High Skill Tech Staff",
    6: "Accountants",
    7: "Medicine Staff",
    8: "Cooking Staff",
    9: "Security Staff",
    10: "Cleaning Staff",
    11: "Private Service Staff",
    12: "Low-skill Laborers",
    13: "Waiters / Barmen Staff",
    14: "Secretaries",
    15: "Realty Agents",
    16: "HR Staff",
    17: "IT Staff",
    18: "Other / Unknown",
}

# Feature importance labels for explanation
FEATURE_LABELS = {
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Car Ownership",
    "FLAG_OWN_REALTY": "Property Ownership",
    "CNT_CHILDREN": "Number of Children",
    "AMT_INCOME_TOTAL": "Annual Income",
    "NAME_INCOME_TYPE": "Income Type",
    "NAME_EDUCATION_TYPE": "Education Level",
    "NAME_FAMILY_STATUS": "Family Status",
    "NAME_HOUSING_TYPE": "Housing Type",
    "OCCUPATION_TYPE": "Occupation",
    "FLAG_WORK_PHONE": "Has Work Phone",
    "FLAG_PHONE": "Has Home Phone",
    "FLAG_EMAIL": "Has Email",
    "CNT_FAM_MEMBERS": "Family Members",
    "AGE_YEARS": "Age",
    "YEARS_EMPLOYED": "Years Employed",
}


def get_decision_strength(prob_unsafe: float) -> dict:
    """Classify the strength of the approval/rejection decision."""
    if prob_unsafe < 0.10:
        return {"label": "Strong Approval", "color": "#16a34a", "emoji": "check",
                "description": "Very low risk — strong candidate for approval."}
    elif prob_unsafe < 0.25:
        return {"label": "Likely Approval", "color": "#65a30d", "emoji": "check",
                "description": "Low risk profile. Approval is likely."}
    elif prob_unsafe < 0.40:
        return {"label": "Leaning Approval", "color": "#ca8a04", "emoji": "minus",
                "description": "Moderate risk. Could go either way, leaning approve."}
    elif prob_unsafe < 0.55:
        return {"label": "Borderline", "color": "#d97706", "emoji": "minus",
                "description": "On the fence — additional review recommended."}
    elif prob_unsafe < 0.70:
        return {"label": "Leaning Rejection", "color": "#ea580c", "emoji": "minus",
                "description": "Elevated risk. Leaning toward rejection."}
    elif prob_unsafe < 0.85:
        return {"label": "Likely Rejection", "color": "#dc2626", "emoji": "x",
                "description": "High risk profile. Rejection is likely."}
    else:
        return {"label": "Strong Rejection", "color": "#991b1b", "emoji": "x",
                "description": "Very high risk — strong candidate for rejection."}


def get_top_reasons(feature_values: dict, prob_unsafe: float) -> list:
    """
    Use the Random Forest's feature importances + the applicant's values
    to produce human-readable reasons for the decision.
    """
    rf_clf = model.named_steps["clf"]
    importances = rf_clf.feature_importances_
    feature_names = FEATURE_ORDER

    # Pair each feature with its importance
    feat_imp = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    reasons = []
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


def _human_readable_value(feat_name: str, val) -> str:
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
    return str(val)


@app.route("/")
def index():
    return render_template(
        "index.html",
        income_types=INCOME_TYPES,
        education_types=EDUCATION_TYPES,
        family_status=FAMILY_STATUS,
        housing_types=HOUSING_TYPES,
        occupation_types=OCCUPATION_TYPES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = {
            "CODE_GENDER": int(data["gender"]),
            "FLAG_OWN_CAR": int(data["own_car"]),
            "FLAG_OWN_REALTY": int(data["own_realty"]),
            "CNT_CHILDREN": int(data["children"]),
            "AMT_INCOME_TOTAL": float(data["income"]),
            "NAME_INCOME_TYPE": int(data["income_type"]),
            "NAME_EDUCATION_TYPE": int(data["education"]),
            "NAME_FAMILY_STATUS": int(data["family_status"]),
            "NAME_HOUSING_TYPE": int(data["housing"]),
            "OCCUPATION_TYPE": int(data["occupation"]),
            "FLAG_WORK_PHONE": int(data["work_phone"]),
            "FLAG_PHONE": int(data["phone"]),
            "FLAG_EMAIL": int(data["email"]),
            "CNT_FAM_MEMBERS": int(data["family_members"]),
            "AGE_YEARS": float(data["age"]),
            "YEARS_EMPLOYED": float(data["years_employed"]),
        }
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X = pd.DataFrame([features], columns=FEATURE_ORDER)
    prob_unsafe = float(model.predict_proba(X)[:, 1][0])
    prediction = int(model.predict(X)[0])
    decision = "Rejected" if prediction == 1 else "Approved"

    strength = get_decision_strength(prob_unsafe)
    reasons = get_top_reasons(features, prob_unsafe)

    return jsonify({
        "decision": decision,
        "probability_risky": round(prob_unsafe * 100, 1),
        "probability_safe": round((1 - prob_unsafe) * 100, 1),
        "strength": strength,
        "top_factors": reasons,
    })


if __name__ == "__main__":
    print("Starting Credit Card Approval app at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
