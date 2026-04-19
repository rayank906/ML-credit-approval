"""
Credit Card Approval — AI Bank Management Platform.
Full-stack FastAPI application with customer + admin experiences.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent
for _path in (_BACKEND_DIR, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fastapi import Depends, FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
try:
    from redis.asyncio import Redis  # type: ignore  # noqa: E402
    from redis.exceptions import RedisError  # type: ignore  # noqa: E402

    _REDIS_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as e:  # pragma: no cover
    Redis = Any  # type: ignore[misc,assignment]

    class RedisError(Exception):
        pass

    _REDIS_IMPORT_ERROR = e
from sqlalchemy import func  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from starlette.templating import Jinja2Templates  # noqa: E402

from auth import create_access_token, decode_token, hash_password, verify_password  # noqa: E402
from database import Application, AuditLog, Draft, Job, Notification, User, get_db, init_db  # noqa: E402
from decision import get_top_reasons  # noqa: E402
from features import (  # noqa: E402
    EDUCATION_TYPES,
    FAMILY_STATUS,
    HOUSING_TYPES,
    INCOME_TYPES,
    OCCUPATION_TYPES,
)
from prediction import model, run_prediction  # noqa: E402
from schemas import (  # noqa: E402
    ApplicationDetail,
    ApplicationSummary,
    AuditLogEntry,
    AuthResponse,
    ChangePasswordRequest,
    CreditBreakdownItem,
    CustomerStats,
    DecisionStrength,
    DraftResponse,
    DraftSave,
    FairnessGroup,
    FairnessMetric,
    FairnessResponse,
    FlagRequest,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    LoginRequest,
    NotificationResponse,
    OverrideRequest,
    PredictRequest,
    PredictResponse,
    RegisterRequest,
    StaffStats,
    ToggleUserActiveRequest,
    TopFactor,
    UpdateProfileRequest,
    UpdateUserRoleRequest,
    UserResponse,
    WhatIfRequest,
    WhatIfResponse,
)

templates = Jinja2Templates(directory=str(_BACKEND_DIR / "templates"))
app = FastAPI(title="Credit Card Approval", version="3.0.0")
app.mount("/static", StaticFiles(directory=str(_BACKEND_DIR / "static")), name="static")

init_db()

logger = logging.getLogger("backend.asgi")

STAFF_ROLES = frozenset({"admin", "officer"})
RATE_LIMIT_PER_DAY = 20
CONFIDENCE_LOW_THRESHOLD = 65.0

JOB_QUEUE_NAME = os.environ.get("JOB_QUEUE_NAME", "ml_jobs")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
JOB_POST_LIMIT_PER_MIN = int(os.environ.get("JOB_POST_LIMIT_PER_MIN", "30"))
JOB_GET_LIMIT_PER_MIN = int(os.environ.get("JOB_GET_LIMIT_PER_MIN", "120"))

_redis: Redis | None = None


def get_redis() -> Redis:
    if _REDIS_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=503,
            detail="Redis is not installed. Install with: pip install redis (or: pip install -r requirements.txt).",
        )
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _stable_json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_job_input_hash(*, user_id: str, payload: dict[str, object]) -> str:
    canonical = _stable_json_dumps(payload)
    return hashlib.sha256(f"{user_id}:{canonical}".encode("utf-8")).hexdigest()


async def _enforce_rate_limit(*, user: User, request: Request, op: str, limit: int) -> None:
    redis = get_redis()
    client_ip = request.client.host if request.client else "unknown"
    key = f"rl:jobs:{op}:{user.id}:{client_ip}"
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 60)
    except RedisError as e:
        logger.warning("Rate limiter unavailable", extra={"key": key}, exc_info=e)
        return
    if count > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry later.")


# ── Auth helpers ──────────────────────────────────────────────


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    payload = decode_token(auth_header[7:])
    if payload is None:
        return None
    return db.query(User).filter(User.email == payload.get("sub")).first()


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")
    return user


def require_staff(request: Request, db: Session = Depends(get_db)) -> User:
    user = require_user(request, db)
    if user.role not in STAFF_ROLES:
        raise HTTPException(status_code=403, detail="Bank staff access required")
    return user


def require_admin(request: Request, db: Session = Depends(get_db)) -> User:
    user = require_user(request, db)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def create_notification(db: Session, user_id, message: str, ntype: str = "info"):
    n = Notification(user_id=user_id, message=message, type=ntype)
    db.add(n)


# ── Exception handlers ────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
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

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception",
        extra={"method": request.method, "path": request.url.path},
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Pages ─────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", context={
        "income_types": INCOME_TYPES, "education_types": EDUCATION_TYPES,
        "family_status": FAMILY_STATUS, "housing_types": HOUSING_TYPES,
        "occupation_types": OCCUPATION_TYPES,
    })

@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"ok": True, "time": datetime.utcnow().isoformat()}


# ── Auth endpoints ────────────────────────────────────────────


@app.post("/auth/register", response_model=AuthResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    role = body.role or "customer"
    if role == "admin":
        expected = os.environ.get("ADMIN_ENROLLMENT_CODE")
        provided = (body.enrollment_code or "").strip()
        if not expected:
            raise HTTPException(status_code=403, detail="Admin signups are disabled")
        if provided != expected:
            raise HTTPException(status_code=403, detail="Invalid admin enrollment code")
    user = User(email=body.email, password_hash=hash_password(body.password), full_name=body.full_name, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    create_notification(db, user.id, f"Welcome to the platform, {user.full_name}!", "success")
    db.commit()
    token = create_access_token({"sub": user.email, "role": user.role})
    return AuthResponse(token=token, email=user.email, full_name=user.full_name, role=user.role)


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")
    token = create_access_token({"sub": user.email, "role": user.role})
    return AuthResponse(token=token, email=user.email, full_name=user.full_name, role=user.role)


@app.get("/auth/me", response_model=UserResponse)
def me(user: User = Depends(require_user)):
    return UserResponse(id=str(user.id), email=user.email, full_name=user.full_name, role=user.role, is_active=user.is_active, created_at=user.created_at)


@app.put("/auth/profile")
def update_profile(body: UpdateProfileRequest, user: User = Depends(require_user), db: Session = Depends(get_db)):
    user.full_name = body.full_name
    db.commit()
    return {"ok": True, "full_name": user.full_name}


@app.put("/auth/password")
def change_password(body: ChangePasswordRequest, user: User = Depends(require_user), db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    return {"ok": True}


# ── Prediction ────────────────────────────────────────────────


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)

    # Rate limiting
    if user:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = db.query(func.count(Application.id)).filter(
            Application.user_id == user.id, Application.created_at >= today_start
        ).scalar() or 0
        if today_count >= RATE_LIMIT_PER_DAY:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Max {RATE_LIMIT_PER_DAY} applications per day.")

    try:
        r = run_prediction(body.model_dump())
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}") from e

    status = "auto_rejected" if r["decision"] == "Rejected" else "auto_approved"

    if user:
        application = Application(
            user_id=user.id, gender=body.gender, own_car=body.own_car,
            own_realty=body.own_realty, children=body.children, income=body.income,
            income_type=body.income_type, education=body.education,
            family_status=body.family_status, housing=body.housing,
            occupation=body.occupation, work_phone=body.work_phone,
            phone=body.phone, email_flag=body.email, family_members=body.family_members,
            age=body.age, years_employed=body.years_employed,
            decision=r["decision"], status=status,
            credit_score=int(r["credit_score"]),
            probability_risky=round(r["blended"] * 100, 1),
            probability_safe=round((1 - r["blended"]) * 100, 1),
            model_probability_risky=round(r["prob_unsafe"] * 100, 1),
            confidence_score=r["confidence"],
            credit_breakdown=[b for b in r["breakdown"]],
        )
        db.add(application)
        msg = f"Application {r['decision'].lower()} — credit score {int(r['credit_score'])}, confidence {r['confidence']}%"
        create_notification(db, user.id, msg, "success" if r["decision"] == "Approved" else "warning")
        db.commit()

    return PredictResponse(
        decision=r["decision"], credit_score=int(r["credit_score"]),
        probability_risky=round(r["blended"] * 100, 1),
        probability_safe=round((1 - r["blended"]) * 100, 1),
        model_probability_risky=round(r["prob_unsafe"] * 100, 1),
        confidence_score=r["confidence"],
        strength=DecisionStrength(**r["strength"]),
        top_factors=[TopFactor(**f) for f in r["reasons"]],
        credit_breakdown=[CreditBreakdownItem(**b) for b in r["breakdown"]],
        risk_tips=r["tips"],
    )


# ── What-If Simulator ────────────────────────────────────────


@app.post("/whatif", response_model=WhatIfResponse)
def whatif(body: WhatIfRequest):
    try:
        r = run_prediction(body.model_dump())
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}") from e
    return WhatIfResponse(
        decision=r["decision"], credit_score=int(r["credit_score"]),
        probability_risky=round(r["blended"] * 100, 1),
        probability_safe=round((1 - r["blended"]) * 100, 1),
        confidence_score=r["confidence"],
        strength_label=r["strength"]["label"], strength_color=r["strength"]["color"],
        strength_description=r["strength"]["description"],
        credit_breakdown=[CreditBreakdownItem(**b) for b in r["breakdown"]],
        risk_tips=r["tips"],
        top_factors=[TopFactor(**f) for f in r["reasons"]],
    )


# ── Customer applications ─────────────────────────────────────


def _app_summary(a: Application, u: User | None = None) -> ApplicationSummary:
    return ApplicationSummary(
        id=str(a.id), decision=a.decision, status=a.status or "auto_approved",
        credit_score=a.credit_score,
        probability_risky=a.probability_risky, probability_safe=a.probability_safe,
        confidence_score=a.confidence_score,
        income=a.income, age=a.age, flagged=a.flagged or False,
        created_at=a.created_at,
        applicant_email=u.email if u else None, applicant_name=u.full_name if u else None,
    )


def _app_detail(a: Application, u: User | None = None) -> ApplicationDetail:
    breakdown = None
    if a.credit_breakdown:
        breakdown = [CreditBreakdownItem(**b) if isinstance(b, dict) else b for b in a.credit_breakdown]
    # Explainability: compute top factors on-demand from stored application features.
    try:
        features = {
            "CODE_GENDER": int(a.gender), "FLAG_OWN_CAR": int(a.own_car),
            "FLAG_OWN_REALTY": int(a.own_realty), "CNT_CHILDREN": int(a.children),
            "AMT_INCOME_TOTAL": float(a.income), "NAME_INCOME_TYPE": int(a.income_type),
            "NAME_EDUCATION_TYPE": int(a.education), "NAME_FAMILY_STATUS": int(a.family_status),
            "NAME_HOUSING_TYPE": int(a.housing), "OCCUPATION_TYPE": int(a.occupation),
            "FLAG_WORK_PHONE": int(a.work_phone), "FLAG_PHONE": int(a.phone),
            "FLAG_EMAIL": int(a.email_flag), "CNT_FAM_MEMBERS": int(a.family_members),
            "AGE_YEARS": float(a.age), "YEARS_EMPLOYED": float(a.years_employed),
            "CREDIT_SCORE": int(a.credit_score),
        }
        top_factors = [TopFactor(**f) for f in get_top_reasons(model, features)]
    except Exception:
        top_factors = None
    return ApplicationDetail(
        id=str(a.id), user_id=str(a.user_id),
        gender=a.gender, own_car=a.own_car, own_realty=a.own_realty,
        children=a.children, income=a.income, income_type=a.income_type,
        education=a.education, family_status=a.family_status, housing=a.housing,
        occupation=a.occupation, work_phone=a.work_phone, phone=a.phone,
        email_flag=a.email_flag, family_members=a.family_members,
        age=a.age, years_employed=a.years_employed,
        decision=a.decision, status=a.status or "auto_approved",
        credit_score=a.credit_score,
        probability_risky=a.probability_risky, probability_safe=a.probability_safe,
        model_probability_risky=a.model_probability_risky,
        confidence_score=a.confidence_score,
        credit_breakdown=breakdown,
        flagged=a.flagged or False, flag_reason=a.flag_reason,
        created_at=a.created_at,
        applicant_email=u.email if u else None, applicant_name=u.full_name if u else None,
        top_factors=top_factors,
    )


@app.get("/applications", response_model=list[ApplicationSummary])
def get_my_applications(user: User = Depends(require_user), db: Session = Depends(get_db)):
    apps = db.query(Application).filter(Application.user_id == user.id).order_by(Application.created_at.desc()).all()
    return [_app_summary(a) for a in apps]


@app.get("/applications/{app_id}", response_model=ApplicationDetail)
def get_application_detail(app_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    a = db.query(Application).filter(Application.id == app_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Application not found")
    if user.role not in STAFF_ROLES and a.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your application")
    owner = db.query(User).filter(User.id == a.user_id).first()
    return _app_detail(a, owner)


@app.get("/stats/me", response_model=CustomerStats)
def customer_stats(user: User = Depends(require_user), db: Session = Depends(get_db)):
    apps = db.query(Application).filter(Application.user_id == user.id).order_by(Application.created_at.desc()).all()
    total = len(apps)
    approved = sum(1 for a in apps if a.status in ("auto_approved", "manually_approved"))
    rejected = sum(1 for a in apps if a.status in ("auto_rejected", "manually_rejected"))
    under_review = sum(1 for a in apps if a.status == "under_review")
    best_score = max((a.credit_score for a in apps), default=None)
    latest = (apps[0].status.replace("_", " ").title()) if apps else None
    return CustomerStats(
        total_applications=total, approved=approved, rejected=rejected,
        under_review=under_review,
        approval_rate=round(approved / total * 100, 1) if total else 0,
        best_credit_score=best_score, latest_decision=latest,
    )


# ── Drafts ────────────────────────────────────────────────────


@app.get("/drafts", response_model=list[DraftResponse])
def list_drafts(user: User = Depends(require_user), db: Session = Depends(get_db)):
    drafts = db.query(Draft).filter(Draft.user_id == user.id).order_by(Draft.updated_at.desc()).all()
    return [DraftResponse(id=str(d.id), name=d.name, form_data=d.form_data, updated_at=d.updated_at, created_at=d.created_at) for d in drafts]


@app.post("/drafts", response_model=DraftResponse)
def save_draft(body: DraftSave, user: User = Depends(require_user), db: Session = Depends(get_db)):
    draft = Draft(user_id=user.id, name=body.name, form_data=body.form_data)
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return DraftResponse(id=str(draft.id), name=draft.name, form_data=draft.form_data, updated_at=draft.updated_at, created_at=draft.created_at)


@app.put("/drafts/{draft_id}", response_model=DraftResponse)
def update_draft(draft_id: str, body: DraftSave, user: User = Depends(require_user), db: Session = Depends(get_db)):
    draft = db.query(Draft).filter(Draft.id == draft_id, Draft.user_id == user.id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    draft.name = body.name
    draft.form_data = body.form_data
    draft.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(draft)
    return DraftResponse(id=str(draft.id), name=draft.name, form_data=draft.form_data, updated_at=draft.updated_at, created_at=draft.created_at)


@app.delete("/drafts/{draft_id}")
def delete_draft(draft_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    draft = db.query(Draft).filter(Draft.id == draft_id, Draft.user_id == user.id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    db.delete(draft)
    db.commit()
    return {"ok": True}


# ── Jobs (Redis queue) ────────────────────────────────────────


async def _try_enqueue_job(*, job_id: str) -> bool:
    try:
        await get_redis().rpush(JOB_QUEUE_NAME, job_id)
        return True
    except RedisError as e:
        logger.error("Failed to enqueue job", extra={"job_id": job_id, "queue": JOB_QUEUE_NAME}, exc_info=e)
        return False


@app.post("/job", response_model=JobSubmitResponse)
async def submit_job(
    body: JobSubmitRequest,
    request: Request,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> JobSubmitResponse:
    await _enforce_rate_limit(user=user, request=request, op="post", limit=JOB_POST_LIMIT_PER_MIN)

    payload_dict = body.payload.model_dump()
    input_hash = _compute_job_input_hash(user_id=str(user.id), payload=payload_dict)

    existing = db.query(Job).filter(Job.user_id == user.id, Job.input_hash == input_hash).first()
    if existing is not None:
        if existing.status == "enqueue_failed":
            if await _try_enqueue_job(job_id=str(existing.id)):
                existing.status = "queued"
                db.commit()
                db.refresh(existing)
        return JobSubmitResponse(job_id=str(existing.id), status=existing.status, deduped=True)

    job = Job(user_id=user.id, input_hash=input_hash, status="queued", request=payload_dict)
    db.add(job)
    db.commit()
    db.refresh(job)

    if not await _try_enqueue_job(job_id=str(job.id)):
        job.status = "enqueue_failed"
        db.commit()
        db.refresh(job)

    return JobSubmitResponse(job_id=str(job.id), status=job.status, deduped=False)


@app.get("/job", response_model=JobStatusResponse)
async def get_job(
    request: Request,
    job_id: str = Query(...),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    await _enforce_rate_limit(user=user, request=request, op="get", limit=JOB_GET_LIMIT_PER_MIN)
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_by_path(job_id: str, request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)) -> JobStatusResponse:
    await _enforce_rate_limit(user=user, request=request, op="get", limit=JOB_GET_LIMIT_PER_MIN)
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ── Notifications ─────────────────────────────────────────────


@app.get("/notifications", response_model=list[NotificationResponse])
def get_notifications(user: User = Depends(require_user), db: Session = Depends(get_db)):
    notifs = db.query(Notification).filter(Notification.user_id == user.id).order_by(Notification.created_at.desc()).limit(50).all()
    return [NotificationResponse(id=str(n.id), message=n.message, type=n.type, read=n.read, created_at=n.created_at) for n in notifs]


@app.post("/notifications/read-all")
def mark_all_read(user: User = Depends(require_user), db: Session = Depends(get_db)):
    db.query(Notification).filter(Notification.user_id == user.id, Notification.read == False).update({"read": True})  # noqa: E712
    db.commit()
    return {"ok": True}


@app.get("/notifications/unread-count")
def unread_count(user: User = Depends(require_user), db: Session = Depends(get_db)):
    count = db.query(func.count(Notification.id)).filter(Notification.user_id == user.id, Notification.read == False).scalar() or 0  # noqa: E712
    return {"count": count}


# ── Admin: applications ───────────────────────────────────────


@app.get("/admin/applications", response_model=list[ApplicationSummary])
def get_all_applications(
    _staff: User = Depends(require_staff), db: Session = Depends(get_db),
    decision: str | None = Query(None), flagged: bool | None = Query(None),
    search: str | None = Query(None), min_risk: float | None = Query(None),
    max_risk: float | None = Query(None), low_confidence: bool | None = Query(None),
):
    q = db.query(Application, User).join(User, Application.user_id == User.id)
    if decision:
        if decision == "Approved":
            q = q.filter(Application.status.in_(["auto_approved", "manually_approved"]))
        elif decision == "Rejected":
            q = q.filter(Application.status.in_(["auto_rejected", "manually_rejected"]))
        elif decision == "under_review":
            q = q.filter(Application.status == "under_review")
    if flagged is not None:
        q = q.filter(Application.flagged == flagged)
    if search:
        like = f"%{search}%"
        q = q.filter((User.full_name.ilike(like)) | (User.email.ilike(like)))
    if min_risk is not None:
        q = q.filter(Application.probability_risky >= min_risk)
    if max_risk is not None:
        q = q.filter(Application.probability_risky <= max_risk)
    if low_confidence:
        q = q.filter(Application.confidence_score < CONFIDENCE_LOW_THRESHOLD)
    rows = q.order_by(Application.created_at.desc()).limit(200).all()
    return [_app_summary(a, u) for a, u in rows]


@app.get("/admin/stats", response_model=StaffStats)
def staff_stats(_staff: User = Depends(require_staff), db: Session = Depends(get_db)):
    total = db.query(func.count(Application.id)).scalar() or 0
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())
    total_today = db.query(func.count(Application.id)).filter(Application.created_at >= today_start).scalar() or 0
    total_this_week = db.query(func.count(Application.id)).filter(Application.created_at >= week_start).scalar() or 0
    approved = db.query(func.count(Application.id)).filter(Application.status.in_(["auto_approved", "manually_approved"])).scalar() or 0
    rejected = db.query(func.count(Application.id)).filter(Application.status.in_(["auto_rejected", "manually_rejected"])).scalar() or 0
    under_review = db.query(func.count(Application.id)).filter(Application.status == "under_review").scalar() or 0
    avg_risk = db.query(func.avg(Application.probability_risky)).scalar() or 0
    flagged_count = db.query(func.count(Application.id)).filter(Application.flagged == True).scalar() or 0  # noqa: E712
    low_conf = db.query(func.count(Application.id)).filter(Application.confidence_score < CONFIDENCE_LOW_THRESHOLD).scalar() or 0
    total_users = db.query(func.count(User.id)).scalar() or 0
    return StaffStats(
        total_applications=total, total_today=total_today, total_this_week=total_this_week,
        approved=approved, rejected=rejected, under_review=under_review,
        approval_rate=round(approved / total * 100, 1) if total else 0,
        avg_risk=round(float(avg_risk), 1), flagged_count=flagged_count,
        low_confidence_count=low_conf, total_users=total_users,
    )


@app.post("/admin/applications/{app_id}/override")
def override_decision(app_id: str, body: OverrideRequest, staff: User = Depends(require_staff), db: Session = Depends(get_db)):
    a = db.query(Application).filter(Application.id == app_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Application not found")
    old_status = a.status
    a.status = body.new_status
    log = AuditLog(application_id=a.id, staff_id=staff.id, action="override", old_status=old_status, new_status=body.new_status, note=body.note)
    db.add(log)
    # Notify the applicant
    status_label = body.new_status.replace("_", " ").title()
    create_notification(db, a.user_id, f"Your application status was changed to: {status_label}. Note: {body.note}", "info")
    db.commit()
    return {"ok": True, "old_status": old_status, "new_status": body.new_status}


@app.post("/admin/applications/{app_id}/flag")
def flag_application(app_id: str, body: FlagRequest, staff: User = Depends(require_staff), db: Session = Depends(get_db)):
    a = db.query(Application).filter(Application.id == app_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Application not found")
    a.flagged = body.flagged
    a.flag_reason = body.reason if body.flagged else None
    log = AuditLog(application_id=a.id, staff_id=staff.id, action="flag" if body.flagged else "unflag", note=body.reason or None)
    db.add(log)
    db.commit()
    return {"ok": True, "flagged": a.flagged}


@app.get("/admin/audit", response_model=list[AuditLogEntry])
def get_audit_log(_staff: User = Depends(require_staff), db: Session = Depends(get_db)):
    rows = db.query(AuditLog, User).join(User, AuditLog.staff_id == User.id).order_by(AuditLog.created_at.desc()).limit(200).all()
    return [AuditLogEntry(id=str(log.id), application_id=str(log.application_id), staff_email=u.email, staff_name=u.full_name, action=log.action, old_status=log.old_status, new_status=log.new_status, note=log.note, created_at=log.created_at) for log, u in rows]


@app.get("/admin/users", response_model=list[UserResponse])
def list_users(_admin: User = Depends(require_staff), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [UserResponse(id=str(u.id), email=u.email, full_name=u.full_name, role=u.role, is_active=u.is_active, created_at=u.created_at) for u in users]


@app.put("/admin/users/{user_id}/role")
def update_user_role(user_id: str, body: UpdateUserRoleRequest, _admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = body.role
    db.commit()
    return {"ok": True, "role": user.role}


@app.put("/admin/users/{user_id}/active")
def toggle_user_active(user_id: str, body: ToggleUserActiveRequest, _admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = body.is_active
    db.commit()
    return {"ok": True, "is_active": user.is_active}


@app.get("/admin/export")
def export_csv(_staff: User = Depends(require_staff), db: Session = Depends(get_db)):
    rows = db.query(Application, User).join(User, Application.user_id == User.id).order_by(Application.created_at.desc()).limit(5000).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "applicant_name", "applicant_email", "decision", "status", "credit_score", "risk_%", "confidence_%", "income", "age", "flagged", "created_at"])
    for a, u in rows:
        writer.writerow([str(a.id), u.full_name, u.email, a.decision, a.status, a.credit_score, a.probability_risky, a.confidence_score, a.income, a.age, a.flagged, a.created_at.isoformat() if a.created_at else ""])
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=applications_export.csv"})


@app.get("/admin/fairness", response_model=FairnessResponse)
def fairness_overview(_staff: User = Depends(require_staff), db: Session = Depends(get_db)):
    rows = db.query(Application).order_by(Application.created_at.desc()).limit(10000).all()

    def outcome(status: str | None) -> str:
        s = status or "auto_approved"
        if s in ("auto_approved", "manually_approved"):
            return "approved"
        if s in ("auto_rejected", "manually_rejected"):
            return "rejected"
        return "under_review"

    def approval_rate(a: int, r: int) -> float:
        denom = a + r
        return round((a / denom) * 100, 1) if denom else 0.0

    approved_total = 0
    rejected_total = 0
    for a in rows:
        o = outcome(a.status)
        if o == "approved":
            approved_total += 1
        elif o == "rejected":
            rejected_total += 1
    overall = approval_rate(approved_total, rejected_total)

    def build_metric(
        key: str,
        label: str,
        group_label_fn,
        group_key_fn,
    ) -> FairnessMetric:
        buckets: dict[str, dict[str, int]] = {}
        for a in rows:
            gk = str(group_key_fn(a))
            if gk not in buckets:
                buckets[gk] = {"approved": 0, "rejected": 0, "under_review": 0}
            o = outcome(a.status)
            buckets[gk][o] += 1

        groups: list[FairnessGroup] = []
        for gk, counts in buckets.items():
            ar = approval_rate(counts["approved"], counts["rejected"])
            groups.append(
                FairnessGroup(
                    group=gk,
                    label=str(group_label_fn(gk)),
                    approved=counts["approved"],
                    rejected=counts["rejected"],
                    under_review=counts["under_review"],
                    approval_rate=ar,
                    delta_vs_overall=round(ar - overall, 1),
                )
            )
        # Stable display ordering where possible (numeric keys, then alpha).
        groups.sort(key=lambda g: (not g.group.isdigit(), int(g.group) if g.group.isdigit() else g.label))
        return FairnessMetric(key=key, label=label, overall_approval_rate=overall, groups=groups)

    def age_bucket(a: Application) -> str:
        age = float(a.age or 0)
        if age < 25:
            return "18-24"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        return "55+"

    metrics = [
        build_metric(
            "gender",
            "Approval Rate by Gender",
            group_label_fn=lambda gk: "Male" if gk == "0" else ("Female" if gk == "1" else "Unknown"),
            group_key_fn=lambda a: a.gender if a.gender is not None else "unknown",
        ),
        build_metric(
            "age_bucket",
            "Approval Rate by Age Bucket",
            group_label_fn=lambda gk: gk,
            group_key_fn=age_bucket,
        ),
        build_metric(
            "education",
            "Approval Rate by Education Level",
            group_label_fn=lambda gk: EDUCATION_TYPES.get(int(gk), f"Education {gk}") if gk.isdigit() else str(gk),
            group_key_fn=lambda a: a.education if a.education is not None else "unknown",
        ),
    ]

    return FairnessResponse(
        generated_at=datetime.utcnow(),
        total_applications=len(rows),
        overall_approval_rate=overall,
        metrics=metrics,
    )


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting Credit Card Approval app at http://127.0.0.1:{port}")
    uvicorn.run(
        "asgi:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        reload_dirs=[str(_BACKEND_DIR)],
        app_dir=str(_BACKEND_DIR),
    )
