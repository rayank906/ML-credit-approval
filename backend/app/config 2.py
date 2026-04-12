import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
    JWT_SECRET = os.environ.get("JWT_SECRET", "jwt-dev-secret-change-in-prod")
    JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))
    DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "data", "bank.db"))
    MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "ml", "credit_approval_model.joblib"))
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "RRn5x6PY3G0mNqEbKz8V7t_2aLX4w1sO9jCfDhWuI0Q=")
    RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "60/minute")
    RATE_LIMIT_LOGIN = os.environ.get("RATE_LIMIT_LOGIN", "5/minute")
    RATE_LIMIT_SUBMIT = os.environ.get("RATE_LIMIT_SUBMIT", "30/minute")
