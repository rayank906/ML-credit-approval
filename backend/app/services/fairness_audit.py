from ..extensions import get_db
from .ml_engine import (
    INCOME_TYPES, EDUCATION_TYPES, FAMILY_STATUS,
    HOUSING_TYPES, OCCUPATION_TYPES,
)

DEMOGRAPHIC_COLUMNS = {
    "gender": {
        "db_col": "code_gender",
        "labels": {0: "Male", 1: "Female"},
    },
    "income_type": {
        "db_col": "name_income_type",
        "labels": INCOME_TYPES,
    },
    "education": {
        "db_col": "name_education_type",
        "labels": EDUCATION_TYPES,
    },
    "family_status": {
        "db_col": "name_family_status",
        "labels": FAMILY_STATUS,
    },
    "housing": {
        "db_col": "name_housing_type",
        "labels": HOUSING_TYPES,
    },
    "occupation": {
        "db_col": "occupation_type",
        "labels": OCCUPATION_TYPES,
    },
    "age_group": {
        "db_col": "age_years",
        "labels": None,
    },
}


def run_audit(demographic):
    config = DEMOGRAPHIC_COLUMNS.get(demographic)
    if not config:
        return None

    db = get_db()

    if demographic == "age_group":
        rows = db.execute("""
            SELECT
                CASE
                    WHEN age_years < 25 THEN '18-24'
                    WHEN age_years < 35 THEN '25-34'
                    WHEN age_years < 45 THEN '35-44'
                    WHEN age_years < 55 THEN '45-54'
                    ELSE '55+'
                END as group_value,
                COUNT(*) as total,
                SUM(CASE WHEN final_decision = 'approved' OR (final_decision IS NULL AND ml_decision = 'approved') THEN 1 ELSE 0 END) as approved,
                AVG(blended_risk_score) as avg_risk,
                AVG(credit_score) as avg_credit_score
            FROM applications
            WHERE status != 'pending'
            GROUP BY group_value
            ORDER BY group_value
        """).fetchall()
    else:
        col = config["db_col"]
        rows = db.execute(f"""
            SELECT
                {col} as group_value,
                COUNT(*) as total,
                SUM(CASE WHEN final_decision = 'approved' OR (final_decision IS NULL AND ml_decision = 'approved') THEN 1 ELSE 0 END) as approved,
                AVG(blended_risk_score) as avg_risk,
                AVG(credit_score) as avg_credit_score
            FROM applications
            WHERE status != 'pending'
            GROUP BY {col}
            ORDER BY {col}
        """).fetchall()

    labels = config["labels"]
    groups = []
    for row in rows:
        gv = row["group_value"]
        group_label = labels.get(int(gv), str(gv)) if labels and gv is not None else str(gv)
        total = row["total"]
        approved = row["approved"]
        groups.append({
            "group": group_label,
            "group_value": gv,
            "total": total,
            "approved": approved,
            "rejected": total - approved,
            "approval_rate": round(approved / total * 100, 1) if total > 0 else 0,
            "avg_risk": round(row["avg_risk"] * 100, 1) if row["avg_risk"] else 0,
            "avg_credit_score": round(row["avg_credit_score"]) if row["avg_credit_score"] else 0,
        })

    return {
        "demographic": demographic,
        "groups": groups,
        "available_demographics": list(DEMOGRAPHIC_COLUMNS.keys()),
    }
