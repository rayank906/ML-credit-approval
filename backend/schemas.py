"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── Auth ──────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=1)
    role: str = Field("customer", pattern="^(customer|admin)$")
    enrollment_code: str | None = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str
    full_name: str
    role: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)


class UpdateProfileRequest(BaseModel):
    full_name: str = Field(..., min_length=1)


class UpdateUserRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(customer|officer|admin)$")


class ToggleUserActiveRequest(BaseModel):
    is_active: bool


# ── Predict ───────────────────────────────────────────────────


class PredictRequest(BaseModel):
    gender: int = Field(..., description="0 = male, 1 = female")
    own_car: int
    own_realty: int
    children: int
    income: float
    income_type: int
    education: int
    family_status: int
    housing: int
    occupation: int
    work_phone: int
    phone: int
    email: int
    family_members: int
    age: float
    years_employed: float


class DecisionStrength(BaseModel):
    label: str
    color: str
    emoji: str
    description: str


class TopFactor(BaseModel):
    feature: str
    value: str
    importance: float


class CreditBreakdownItem(BaseModel):
    factor: str
    points: int
    direction: str  # "positive", "negative", "neutral"


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    decision: str
    credit_score: int
    probability_risky: float
    probability_safe: float
    model_probability_risky: float
    confidence_score: float
    strength: DecisionStrength
    top_factors: list[TopFactor]
    credit_breakdown: list[CreditBreakdownItem]
    risk_tips: list[str]


# ── Jobs (Redis queue) ────────────────────────────────────────


class JobSubmitRequest(BaseModel):
    payload: PredictRequest = Field(..., description="Prediction input; validated at submit time.")


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    deduped: bool = Field(False, description="True when an identical job already existed.")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Any | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


# ── What-If Simulator ────────────────────────────────────────


class WhatIfRequest(BaseModel):
    gender: int
    own_car: int
    own_realty: int
    children: int
    income: float
    income_type: int
    education: int
    family_status: int
    housing: int
    occupation: int
    work_phone: int
    phone: int
    email: int
    family_members: int
    age: float
    years_employed: float


class WhatIfResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    decision: str
    credit_score: int
    probability_risky: float
    probability_safe: float
    confidence_score: float
    strength_label: str
    strength_color: str
    strength_description: str
    credit_breakdown: list[CreditBreakdownItem]
    risk_tips: list[str]
    top_factors: list[TopFactor]


# ── Application ───────────────────────────────────────────────


class ApplicationSummary(BaseModel):
    id: str
    decision: str
    status: str
    credit_score: int
    probability_risky: float
    probability_safe: float
    confidence_score: float | None
    income: float
    age: float
    flagged: bool
    created_at: datetime
    applicant_email: str | None = None
    applicant_name: str | None = None


class ApplicationDetail(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    user_id: str
    gender: int
    own_car: int
    own_realty: int
    children: int
    income: float
    income_type: int
    education: int
    family_status: int
    housing: int
    occupation: int
    work_phone: int
    phone: int
    email_flag: int
    family_members: int
    age: float
    years_employed: float
    decision: str
    status: str
    credit_score: int
    probability_risky: float
    probability_safe: float
    model_probability_risky: float
    confidence_score: float | None
    credit_breakdown: list[CreditBreakdownItem] | None
    flagged: bool
    flag_reason: str | None
    created_at: datetime
    applicant_email: str | None = None
    applicant_name: str | None = None
    top_factors: list[TopFactor] | None = None


# ── Fairness / Bias ───────────────────────────────────────────


class FairnessGroup(BaseModel):
    group: str
    label: str
    approved: int
    rejected: int
    under_review: int
    approval_rate: float
    delta_vs_overall: float


class FairnessMetric(BaseModel):
    key: str
    label: str
    overall_approval_rate: float
    groups: list[FairnessGroup]


class FairnessResponse(BaseModel):
    generated_at: datetime
    total_applications: int
    overall_approval_rate: float
    metrics: list[FairnessMetric]


# ── Override / Flag ───────────────────────────────────────────


class OverrideRequest(BaseModel):
    new_status: str = Field(
        ..., pattern="^(manually_approved|manually_rejected|under_review)$"
    )
    note: str = Field(..., min_length=1, max_length=500)


class FlagRequest(BaseModel):
    flagged: bool
    reason: str = Field("", max_length=500)


# ── Audit Log ────────────────────────────────────────────────


class AuditLogEntry(BaseModel):
    id: str
    application_id: str
    staff_email: str
    staff_name: str
    action: str
    old_status: str | None
    new_status: str | None
    note: str | None
    created_at: datetime


# ── Stats ─────────────────────────────────────────────────────


class CustomerStats(BaseModel):
    total_applications: int
    approved: int
    rejected: int
    under_review: int
    approval_rate: float
    best_credit_score: int | None
    latest_decision: str | None


class StaffStats(BaseModel):
    total_applications: int
    total_today: int
    total_this_week: int
    approved: int
    rejected: int
    under_review: int
    approval_rate: float
    avg_risk: float
    flagged_count: int
    low_confidence_count: int
    total_users: int


# ── Drafts ────────────────────────────────────────────────────


class DraftSave(BaseModel):
    name: str = Field("Untitled Draft", max_length=100)
    form_data: dict[str, Any]


class DraftResponse(BaseModel):
    id: str
    name: str
    form_data: dict[str, Any]
    updated_at: datetime
    created_at: datetime


# ── Notifications ─────────────────────────────────────────────


class NotificationResponse(BaseModel):
    id: str
    message: str
    type: str
    read: bool
    created_at: datetime
