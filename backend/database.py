"""Database connection and table management using SQLAlchemy + Supabase PostgreSQL."""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"options": "-c statement_timeout=30000"},
    execution_options={"prepared_statement_cache_size": 0},
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ── ORM Models ───────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(String, nullable=False, default="customer")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Application(Base):
    __tablename__ = "applications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Input fields
    gender = Column(Integer)
    own_car = Column(Integer)
    own_realty = Column(Integer)
    children = Column(Integer)
    income = Column(Float)
    income_type = Column(Integer)
    education = Column(Integer)
    family_status = Column(Integer)
    housing = Column(Integer)
    occupation = Column(Integer)
    work_phone = Column(Integer)
    phone = Column(Integer)
    email_flag = Column(Integer)
    family_members = Column(Integer)
    age = Column(Float)
    years_employed = Column(Float)

    # Prediction results
    decision = Column(String)
    credit_score = Column(Integer)
    probability_risky = Column(Float)
    probability_safe = Column(Float)
    model_probability_risky = Column(Float)
    confidence_score = Column(Float, nullable=True)
    credit_breakdown = Column(JSON, nullable=True)

    # Status tracking
    status = Column(String, nullable=False, default="auto_approved")
    flagged = Column(Boolean, nullable=False, default=False)
    flag_reason = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class Draft(Base):
    __tablename__ = "drafts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    form_data = Column(JSON, nullable=False)
    name = Column(String, nullable=False, default="Untitled Draft")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    message = Column(String, nullable=False)
    type = Column(String, nullable=False, default="info")  # info, success, warning
    read = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id"), nullable=False)
    staff_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    old_status = Column(String, nullable=True)
    new_status = Column(String, nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("user_id", "input_hash", name="uq_jobs_user_input_hash"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    input_hash = Column(String, nullable=False)

    status = Column(String, nullable=False, default="queued")  # queued|processing|succeeded|failed|enqueue_failed
    request = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── Helpers ───────────────────────────────────────────────────


def init_db() -> None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[WARNING] Could not create tables (DB may be unreachable): {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
