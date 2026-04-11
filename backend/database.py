"""Database connection and table management using SQLAlchemy + Supabase PostgreSQL."""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")

# Transaction pooler (PgBouncer) doesn't support prepared statements
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
    role = Column(String, nullable=False, default="customer")  # "customer" or "admin"
    created_at = Column(DateTime, default=datetime.utcnow)


class Application(Base):
    __tablename__ = "applications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # ── Input fields ──
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

    # ── Prediction results ──
    decision = Column(String)
    credit_score = Column(Integer)
    probability_risky = Column(Float)
    probability_safe = Column(Float)
    model_probability_risky = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)


# ── Helpers ───────────────────────────────────────────────────


def init_db() -> None:
    """Create all tables if they don't already exist."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[WARNING] Could not create tables (DB may be unreachable): {e}")


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
