"""
FastAPI web application for Credit Card Approval predictions.
Loads the trained Random Forest model and serves a questionnaire UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent
for _path in (_BACKEND_DIR, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.templating import Jinja2Templates  # noqa: E402

from credit_score import compute_credit_score  # noqa: E402
from decision import blend_risk, get_decision_strength, get_top_reasons  # noqa: E402
from features import (  # noqa: E402
    EDUCATION_TYPES,
    FAMILY_STATUS,
    FEATURE_ORDER,
    HOUSING_TYPES,
    INCOME_TYPES,
    OCCUPATION_TYPES,
)
from schemas import DecisionStrength, PredictRequest, PredictResponse, TopFactor  # noqa: E402

_MODEL_PATH = _REPO_ROOT / "model" / "credit_approval_model.joblib"

model = joblib.load(_MODEL_PATH)

templates = Jinja2Templates(directory=str(_BACKEND_DIR / "templates"))
app = FastAPI(title="Credit Card Approval", version="1.0.0")
app.mount(
    "/static",
    StaticFiles(directory=str(_BACKEND_DIR / "static")),
    name="static",
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    if errors:
        first = errors[0]
        msg = first.get("msg", "Invalid input")
        loc = " -> ".join(str(x) for x in first.get("loc", ()))
        detail = f"{loc}: {msg}" if loc else msg
    else:
        detail = "Invalid input"
    return JSONResponse(status_code=400, content={"error": f"Invalid input: {detail}"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "income_types": INCOME_TYPES,
            "education_types": EDUCATION_TYPES,
            "family_status": FAMILY_STATUS,
            "housing_types": HOUSING_TYPES,
            "occupation_types": OCCUPATION_TYPES,
        },
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    try:
        features = {
            "CODE_GENDER": int(body.gender),
            "FLAG_OWN_CAR": int(body.own_car),
            "FLAG_OWN_REALTY": int(body.own_realty),
            "CNT_CHILDREN": int(body.children),
            "AMT_INCOME_TOTAL": float(body.income),
            "NAME_INCOME_TYPE": int(body.income_type),
            "NAME_EDUCATION_TYPE": int(body.education),
            "NAME_FAMILY_STATUS": int(body.family_status),
            "NAME_HOUSING_TYPE": int(body.housing),
            "OCCUPATION_TYPE": int(body.occupation),
            "FLAG_WORK_PHONE": int(body.work_phone),
            "FLAG_PHONE": int(body.phone),
            "FLAG_EMAIL": int(body.email),
            "CNT_FAM_MEMBERS": int(body.family_members),
            "AGE_YEARS": float(body.age),
            "YEARS_EMPLOYED": float(body.years_employed),
        }
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}") from e

    credit_score = compute_credit_score(features)
    features["CREDIT_SCORE"] = credit_score
    X = pd.DataFrame([features], columns=FEATURE_ORDER)
    prob_unsafe = float(model.predict_proba(X)[:, 1][0])
    blended_prob_unsafe = blend_risk(prob_unsafe, credit_score)
    decision = "Rejected" if blended_prob_unsafe >= 0.4 else "Approved"

    strength = get_decision_strength(blended_prob_unsafe)
    reasons = get_top_reasons(model, features)

    return PredictResponse(
        decision=decision,
        credit_score=int(credit_score),
        probability_risky=round(blended_prob_unsafe * 100, 1),
        probability_safe=round((1 - blended_prob_unsafe) * 100, 1),
        model_probability_risky=round(prob_unsafe * 100, 1),
        strength=DecisionStrength(**strength),
        top_factors=[TopFactor(**r) for r in reasons],
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, str):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn

    print("Starting Credit Card Approval app at http://127.0.0.1:8000")
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(_BACKEND_DIR)],
    )
