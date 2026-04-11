"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str
    full_name: str
    role: str


class UserResponse(BaseModel):
    email: str
    full_name: str
    role: str


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


class PredictResponse(BaseModel):
    decision: str
    credit_score: int
    probability_risky: float
    probability_safe: float
    model_probability_risky: float
    strength: DecisionStrength
    top_factors: list[TopFactor]


# ── Application History ───────────────────────────────────────


class ApplicationResponse(BaseModel):
    id: str
    decision: str
    credit_score: int
    probability_risky: float
    probability_safe: float
    income: float
    age: float
    created_at: datetime
