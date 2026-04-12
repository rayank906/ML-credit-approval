import json

from ..extensions import get_db

FEATURE_COLUMNS = [
    "code_gender", "flag_own_car", "flag_own_realty", "cnt_children",
    "amt_income_total", "name_income_type", "name_education_type",
    "name_family_status", "name_housing_type", "occupation_type",
    "flag_work_phone", "flag_phone", "flag_email", "cnt_fam_members",
    "age_years", "years_employed",
]


def create_application(user_id, cleaned_data, ml_results):
    db = get_db()
    cursor = db.execute(
        """INSERT INTO applications (
            user_id, status,
            code_gender, flag_own_car, flag_own_realty, cnt_children,
            amt_income_total, name_income_type, name_education_type,
            name_family_status, name_housing_type, occupation_type,
            flag_work_phone, flag_phone, flag_email, cnt_fam_members,
            age_years, years_employed,
            credit_score, ml_probability_risky, blended_risk_score,
            ml_decision, ml_strength_label, final_decision,
            recommended_credit_limit, top_factors_json, model_version
        ) VALUES (?, 'ml_reviewed',
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            cleaned_data["gender"], cleaned_data["own_car"],
            cleaned_data["own_realty"], cleaned_data["children"],
            cleaned_data["income"], cleaned_data["income_type"],
            cleaned_data["education"], cleaned_data["family_status"],
            cleaned_data["housing"], cleaned_data["occupation"],
            cleaned_data["work_phone"], cleaned_data["phone"],
            cleaned_data["email"], cleaned_data["family_members"],
            cleaned_data["age"], cleaned_data["years_employed"],
            ml_results["credit_score"], ml_results["probability_risky"],
            ml_results["blended_risk_score"], ml_results["decision"],
            ml_results["strength"]["label"], ml_results["decision"],
            ml_results["recommended_credit_limit"],
            json.dumps(ml_results["top_factors"]),
            ml_results["model_version"],
        ),
    )
    db.commit()
    return get_application(cursor.lastrowid)


def get_application(app_id):
    db = get_db()
    row = db.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_applications_for_user(user_id, status=None, page=1, per_page=20):
    db = get_db()
    params = [user_id]
    query = "SELECT * FROM applications WHERE user_id = ?"
    count_query = "SELECT COUNT(*) FROM applications WHERE user_id = ?"

    if status:
        query += " AND status = ?"
        count_query += " AND status = ?"
        params.append(status)

    total = db.execute(count_query, params).fetchone()[0]
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([per_page, (page - 1) * per_page])
    rows = db.execute(query, params).fetchall()

    return {
        "applications": [_row_to_dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }


def get_all_applications(status=None, sort="created_at_desc", page=1, per_page=20):
    db = get_db()
    params = []
    query = "SELECT * FROM applications"
    count_query = "SELECT COUNT(*) FROM applications"

    if status:
        query += " WHERE status = ?"
        count_query += " WHERE status = ?"
        params.append(status)

    sort_map = {
        "created_at_desc": "created_at DESC",
        "created_at_asc": "created_at ASC",
        "risk_desc": "blended_risk_score DESC",
        "risk_asc": "blended_risk_score ASC",
        "credit_score_desc": "credit_score DESC",
        "credit_score_asc": "credit_score ASC",
    }
    query += f" ORDER BY {sort_map.get(sort, 'created_at DESC')}"

    total = db.execute(count_query, params).fetchone()[0]
    query += " LIMIT ? OFFSET ?"
    params.extend([per_page, (page - 1) * per_page])
    rows = db.execute(query, params).fetchall()

    return {
        "applications": [_row_to_dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }


def update_application(app_id, **kwargs):
    db = get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [app_id]
    db.execute(f"UPDATE applications SET {sets}, updated_at = datetime('now') WHERE id = ?", vals)
    db.commit()
    return get_application(app_id)


def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    if d.get("top_factors_json"):
        try:
            d["top_factors"] = json.loads(d["top_factors_json"])
        except (json.JSONDecodeError, TypeError):
            d["top_factors"] = []
    else:
        d["top_factors"] = []
    return d
