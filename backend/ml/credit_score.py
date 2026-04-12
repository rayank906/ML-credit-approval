from __future__ import annotations

from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd

SCORE_MIN = 300
SCORE_MAX = 850

def _clip_score(score: Union[int, float]) -> int:
    return int(np.clip(round(float(score)), SCORE_MIN, SCORE_MAX))


def _score_income(income: float) -> int:
    if income < 25000:
        return -80
    if income < 50000:
        return -40
    if income < 100000:
        return 20
    if income < 200000:
        return 60
    return 80


def _score_age(age_years: float) -> int:
    if age_years < 25:
        return -60
    if age_years < 35:
        return -20
    if age_years < 45:
        return 20
    if age_years < 55:
        return 40
    return 20


def _score_employment(years_employed: float) -> int:
    if years_employed < 1:
        return -60
    if years_employed < 3:
        return -20
    if years_employed < 7:
        return 20
    return 40


def _score_children(num_children: int) -> int:
    if num_children <= 0:
        return 20
    if num_children <= 2:
        return 0
    return -20


def _score_family_members(num_members: int) -> int:
    return -20 if num_members > 4 else 0


def _score_education(education_type: int) -> int:
    # Encodings from cleaning.py
    if education_type == 4:  # Academic degree
        return 20
    if education_type == 3:  # Higher education
        return 20
    if education_type == 2:  # Incomplete higher
        return 10
    if education_type == 0:  # Lower secondary
        return -20
    return 0


def _score_income_type(income_type: int) -> int:
    # Encodings from cleaning.py
    if income_type == 4:  # Student
        return -30
    if income_type == 2:  # Pensioner
        return 10
    if income_type == 3:  # State servant
        return 10
    if income_type == 1:  # Commercial associate
        return 5
    return 0  # Working


def _score_housing(housing_type: int) -> int:
    # Encodings from cleaning.py
    if housing_type == 4:  # House / apartment (owned)
        return 20
    if housing_type == 1:  # Rented apartment
        return -10
    if housing_type == 0:  # With parents
        return -5
    if housing_type == 5:  # Office apartment
        return -10
    return 0


def _score_occupation(occupation_type: int) -> int:
    # Encodings from cleaning.py
    if occupation_type == 3:  # Managers
        return 15
    if occupation_type == 5:  # High skill tech staff
        return 10
    if occupation_type == 12:  # Low-skill laborers
        return -10
    if occupation_type == 18:  # Unknown
        return -5
    return 0


def compute_credit_score(features: Dict[str, Union[int, float]]) -> int:
    """
    Compute a proxy credit score (300-850) from non-leaky applicant features.
    This is a deterministic, explainable heuristic to weight acceptance decisions.
    """
    base = 600
    score = base
    score += _score_income(float(features.get("AMT_INCOME_TOTAL", 0)))
    score += _score_age(float(features.get("AGE_YEARS", 0)))
    score += _score_employment(float(features.get("YEARS_EMPLOYED", 0)))
    score += 30 if int(features.get("FLAG_OWN_REALTY", 0)) == 1 else 0
    score += 10 if int(features.get("FLAG_OWN_CAR", 0)) == 1 else 0
    score += _score_children(int(features.get("CNT_CHILDREN", 0)))
    score += _score_family_members(int(features.get("CNT_FAM_MEMBERS", 0)))
    score += _score_education(int(features.get("NAME_EDUCATION_TYPE", 1)))
    score += _score_income_type(int(features.get("NAME_INCOME_TYPE", 0)))
    score += _score_housing(int(features.get("NAME_HOUSING_TYPE", 0)))
    score += _score_occupation(int(features.get("OCCUPATION_TYPE", 18)))
    return _clip_score(score)


def add_credit_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with CREDIT_SCORE column added.
    Assumes columns are already encoded per cleaning.py.
    """
    required = [
        "AMT_INCOME_TOTAL",
        "AGE_YEARS",
        "YEARS_EMPLOYED",
        "FLAG_OWN_REALTY",
        "FLAG_OWN_CAR",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for credit score: {missing}")

    out = df.copy()
    out["CREDIT_SCORE"] = (
        600
        + out["AMT_INCOME_TOTAL"].apply(_score_income)
        + out["AGE_YEARS"].apply(_score_age)
        + out["YEARS_EMPLOYED"].apply(_score_employment)
        + out["FLAG_OWN_REALTY"].apply(lambda v: 30 if int(v) == 1 else 0)
        + out["FLAG_OWN_CAR"].apply(lambda v: 10 if int(v) == 1 else 0)
        + out["CNT_CHILDREN"].apply(_score_children)
        + out["CNT_FAM_MEMBERS"].apply(_score_family_members)
        + out["NAME_EDUCATION_TYPE"].apply(_score_education)
        + out["NAME_INCOME_TYPE"].apply(_score_income_type)
        + out["NAME_HOUSING_TYPE"].apply(_score_housing)
        + out["OCCUPATION_TYPE"].apply(_score_occupation)
    ).apply(_clip_score)
    return out
