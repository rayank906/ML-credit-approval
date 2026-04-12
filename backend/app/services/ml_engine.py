import hashlib
import os

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml"))
from credit_score import compute_credit_score


FEATURE_ORDER = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS",
    "AGE_YEARS", "YEARS_EMPLOYED",
    "CREDIT_SCORE",
]

INCOME_TYPES = {
    0: "Working", 1: "Commercial Associate", 2: "Pensioner",
    3: "State Servant", 4: "Student",
}
EDUCATION_TYPES = {
    0: "Lower Secondary", 1: "Secondary / Special", 2: "Incomplete Higher",
    3: "Higher Education", 4: "Academic Degree",
}
FAMILY_STATUS = {
    0: "Single / Not Married", 1: "Married", 2: "Separated",
    3: "Civil Marriage", 4: "Widow",
}
HOUSING_TYPES = {
    0: "With Parents", 1: "Rented Apartment", 2: "Municipal Apartment",
    3: "Co-op Apartment", 4: "House / Apartment (owned)", 5: "Office Apartment",
}
OCCUPATION_TYPES = {
    0: "Laborers", 1: "Core Staff", 2: "Sales Staff", 3: "Managers",
    4: "Drivers", 5: "High Skill Tech Staff", 6: "Accountants",
    7: "Medicine Staff", 8: "Cooking Staff", 9: "Security Staff",
    10: "Cleaning Staff", 11: "Private Service Staff", 12: "Low-skill Laborers",
    13: "Waiters / Barmen Staff", 14: "Secretaries", 15: "Realty Agents",
    16: "HR Staff", 17: "IT Staff", 18: "Other / Unknown",
}
FEATURE_LABELS = {
    "CODE_GENDER": "Gender", "FLAG_OWN_CAR": "Car Ownership",
    "FLAG_OWN_REALTY": "Property Ownership", "CNT_CHILDREN": "Number of Children",
    "AMT_INCOME_TOTAL": "Annual Income", "NAME_INCOME_TYPE": "Income Type",
    "NAME_EDUCATION_TYPE": "Education Level", "NAME_FAMILY_STATUS": "Family Status",
    "NAME_HOUSING_TYPE": "Housing Type", "OCCUPATION_TYPE": "Occupation",
    "FLAG_WORK_PHONE": "Has Work Phone", "FLAG_PHONE": "Has Home Phone",
    "FLAG_EMAIL": "Has Email", "CNT_FAM_MEMBERS": "Family Members",
    "AGE_YEARS": "Age", "YEARS_EMPLOYED": "Years Employed",
    "CREDIT_SCORE": "Credit Score",
}

ALL_MAPPINGS = {
    "income_types": INCOME_TYPES,
    "education_types": EDUCATION_TYPES,
    "family_status": FAMILY_STATUS,
    "housing_types": HOUSING_TYPES,
    "occupation_types": OCCUPATION_TYPES,
}


class MLEngine:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.model_version = self._compute_model_hash(model_path)

    def _compute_model_hash(self, path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:12]

    def predict(self, raw_data):
        features = {
            "CODE_GENDER": int(raw_data["gender"]),
            "FLAG_OWN_CAR": int(raw_data["own_car"]),
            "FLAG_OWN_REALTY": int(raw_data["own_realty"]),
            "CNT_CHILDREN": int(raw_data["children"]),
            "AMT_INCOME_TOTAL": float(raw_data["income"]),
            "NAME_INCOME_TYPE": int(raw_data["income_type"]),
            "NAME_EDUCATION_TYPE": int(raw_data["education"]),
            "NAME_FAMILY_STATUS": int(raw_data["family_status"]),
            "NAME_HOUSING_TYPE": int(raw_data["housing"]),
            "OCCUPATION_TYPE": int(raw_data["occupation"]),
            "FLAG_WORK_PHONE": int(raw_data["work_phone"]),
            "FLAG_PHONE": int(raw_data["phone"]),
            "FLAG_EMAIL": int(raw_data["email"]),
            "CNT_FAM_MEMBERS": int(raw_data["family_members"]),
            "AGE_YEARS": float(raw_data["age"]),
            "YEARS_EMPLOYED": float(raw_data["years_employed"]),
        }

        credit_score = compute_credit_score(features)
        features["CREDIT_SCORE"] = credit_score

        X = pd.DataFrame([features], columns=FEATURE_ORDER)
        prob_unsafe = float(self.model.predict_proba(X)[:, 1][0])
        blended = self.blend_risk(prob_unsafe, credit_score)
        decision = "rejected" if blended >= 0.4 else "approved"
        strength = self.get_decision_strength(blended)
        reasons = self.get_top_reasons(features)
        from .credit_limit import recommend_credit_limit
        credit_limit = recommend_credit_limit(blended, features["AMT_INCOME_TOTAL"], credit_score)

        return {
            "decision": decision,
            "credit_score": int(credit_score),
            "probability_risky": round(blended * 100, 1),
            "probability_safe": round((1 - blended) * 100, 1),
            "model_probability_risky": round(prob_unsafe * 100, 1),
            "strength": strength,
            "top_factors": reasons,
            "recommended_credit_limit": credit_limit,
            "model_version": self.model_version,
            "blended_risk_score": round(blended, 4),
        }

    @staticmethod
    def blend_risk(prob_unsafe, credit_score, weight=0.65):
        credit_norm = (credit_score - 300) / 550
        credit_norm = max(0.0, min(1.0, credit_norm))
        return (weight * (1 - credit_norm)) + ((1 - weight) * prob_unsafe)

    @staticmethod
    def get_decision_strength(prob_unsafe):
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

    def get_top_reasons(self, feature_values):
        rf_clf = self.model.named_steps["clf"]
        importances = rf_clf.feature_importances_
        feat_imp = sorted(zip(FEATURE_ORDER, importances), key=lambda x: x[1], reverse=True)

        reasons = []
        for feat_name, importance in feat_imp[:5]:
            val = feature_values[feat_name]
            label = FEATURE_LABELS.get(feat_name, feat_name)
            human_val = self._human_readable_value(feat_name, val)
            reasons.append({
                "feature": label,
                "value": human_val,
                "importance": round(float(importance) * 100, 1),
            })
        return reasons

    @staticmethod
    def _human_readable_value(feat_name, val):
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
        if feat_name in ("AGE_YEARS", "YEARS_EMPLOYED"):
            return f"{val:.1f} years"
        if feat_name == "CREDIT_SCORE":
            return f"{int(val)}"
        return str(val)
