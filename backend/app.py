"""
FastAPI web application for Credit Card Approval predictions.
Loads the trained Random Forest model and serves a questionnaire UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent
for _path in (_BACKEND_DIR, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from starlette.templating import Jinja2Templates  # noqa: E402

from auth import (  # noqa: E402
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from credit_score import compute_credit_score  # noqa: E402
from database import Application, User, get_db, init_db  # noqa: E402
from decision import blend_risk, get_decision_strength, get_top_reasons  # noqa: E402
from features import (  # noqa: E402
    EDUCATION_TYPES,
    FAMILY_STATUS,
    FEATURE_ORDER,
    HOUSING_TYPES,
    INCOME_TYPES,
    OCCUPATION_TYPES,
)
from schemas import (  # noqa: E402
    ApplicationResponse,
    AuthResponse,
    DecisionStrength,
    LoginRequest,
    PredictRequest,
    PredictResponse,
    RegisterRequest,
    TopFactor,
    UserResponse,
)

_MODEL_PATH = _REPO_ROOT / "model" / "credit_approval_model.joblib"

model = joblib.load(_MODEL_PATH)

templates = Jinja2Templates(directory=str(_BACKEND_DIR / "templates"))
app = FastAPI(title="Credit Card Approval", version="1.0.0")
app.mount(
    "/static",
    StaticFiles(directory=str(_BACKEND_DIR / "static")),
    name="static",
)

# Create database tables on startup
init_db()


# ── Helpers ───────────────────────────────────────────────────


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Extract the logged-in user from the Authorization header (optional)."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    payload = decode_token(token)
    if payload is None:
        return None
    user = db.query(User).filter(User.email == payload.get("sub")).first()
    return user


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Dependency that requires a valid logged-in user."""
    user = get_current_user(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(request: Request, db: Session = Depends(get_db)) -> User:
    """Dependency that requires an admin user."""
    user = require_user(request, db)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ── Exception handlers ────────────────────────────────────────


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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, str):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# ── Pages ─────────────────────────────────────────────────────


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


# ── Auth endpoints ────────────────────────────────────────────


@app.post("/auth/register", response_model=AuthResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        full_name=body.full_name,
        role="customer",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.email, "role": user.role})
    return AuthResponse(
        token=token, email=user.email, full_name=user.full_name, role=user.role
    )


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user.email, "role": user.role})
    return AuthResponse(
        token=token, email=user.email, full_name=user.full_name, role=user.role
    )


@app.get("/auth/me", response_model=UserResponse)
def me(user: User = Depends(require_user)) -> UserResponse:
    return UserResponse(email=user.email, full_name=user.full_name, role=user.role)


# ── Prediction ────────────────────────────────────────────────


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictResponse:
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

    # Save application to DB if user is logged in
    user = get_current_user(request, db)
    if user is not None:
        application = Application(
            user_id=user.id,
            gender=body.gender,
            own_car=body.own_car,
            own_realty=body.own_realty,
            children=body.children,
            income=body.income,
            income_type=body.income_type,
            education=body.education,
            family_status=body.family_status,
            housing=body.housing,
            occupation=body.occupation,
            work_phone=body.work_phone,
            phone=body.phone,
            email_flag=body.email,
            family_members=body.family_members,
            age=body.age,
            years_employed=body.years_employed,
            decision=decision,
            credit_score=int(credit_score),
            probability_risky=round(blended_prob_unsafe * 100, 1),
            probability_safe=round((1 - blended_prob_unsafe) * 100, 1),
            model_probability_risky=round(prob_unsafe * 100, 1),
        )
        db.add(application)
        db.commit()

    return PredictResponse(
        decision=decision,
        credit_score=int(credit_score),
        probability_risky=round(blended_prob_unsafe * 100, 1),
        probability_safe=round((1 - blended_prob_unsafe) * 100, 1),
        model_probability_risky=round(prob_unsafe * 100, 1),
        strength=DecisionStrength(**strength),
        top_factors=[TopFactor(**r) for r in reasons],
    )


# ── Application History ───────────────────────────────────────


@app.get("/applications", response_model=list[ApplicationResponse])
def get_my_applications(user: User = Depends(require_user), db: Session = Depends(get_db)):
    apps = (
        db.query(Application)
        .filter(Application.user_id == user.id)
        .order_by(Application.created_at.desc())
        .all()
    )
    return [
        ApplicationResponse(
            id=str(a.id),
            decision=a.decision,
            credit_score=a.credit_score,
            probability_risky=a.probability_risky,
            probability_safe=a.probability_safe,
            income=a.income,
            age=a.age,
            created_at=a.created_at,
        )
        for a in apps
    ]


@app.get("/admin/applications", response_model=list[ApplicationResponse])
def get_all_applications(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    apps = (
        db.query(Application)
        .order_by(Application.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        ApplicationResponse(
            id=str(a.id),
            decision=a.decision,
            credit_score=a.credit_score,
            probability_risky=a.probability_risky,
            probability_safe=a.probability_safe,
            income=a.income,
            age=a.age,
            created_at=a.created_at,
        )
        for a in apps
    ]


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
