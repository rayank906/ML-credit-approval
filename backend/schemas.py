"""Pydantic request/response models for the HTTP API."""

from pydantic import BaseModel, Field


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
