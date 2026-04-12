from datetime import datetime, timedelta, timezone

import jwt
from flask import Blueprint, current_app, jsonify, request

from ..models.user import create_user, get_user_by_email, verify_password, user_to_dict
from ..extensions import get_db

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def _make_token(user):
    payload = {
        "user_id": user["id"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=current_app.config["JWT_EXPIRY_HOURS"]),
    }
    return jwt.encode(payload, current_app.config["JWT_SECRET"], algorithm="HS256")


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    full_name = data.get("full_name", "").strip()
    role = data.get("role", "applicant")

    if not email or not password or not full_name:
        return jsonify({"error": "Email, password, and full name are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if role not in ("applicant", "officer", "admin"):
        return jsonify({"error": "Invalid role"}), 400

    existing = get_user_by_email(email)
    if existing:
        return jsonify({"error": "Email already registered"}), 409

    user = create_user(email, password, full_name, role)
    token = _make_token(user)

    return jsonify({"user": user_to_dict(user), "token": token}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db = get_db()
    rows = db.execute("SELECT * FROM users WHERE is_active = 1").fetchall()
    matched_user = None
    from ..services.encryption import decrypt_pii
    for row in rows:
        u = dict(row)
        if decrypt_pii(u["email"]) == email:
            matched_user = u
            break

    if not matched_user or not verify_password(matched_user["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    from ..models.user import _decrypt_user, user_to_dict as utd
    user = _decrypt_user(dict(matched_user))
    token = _make_token(user)

    return jsonify({"user": utd(user), "token": token}), 200


@auth_bp.route("/me", methods=["GET"])
def me():
    from ..middleware.auth_middleware import require_auth
    @require_auth
    def _inner():
        from ..models.user import user_to_dict
        from flask import g
        return jsonify({"user": user_to_dict(g.current_user)}), 200
    return _inner()


@auth_bp.route("/me", methods=["PUT"])
def update_me():
    from ..middleware.auth_middleware import require_auth
    @require_auth
    def _inner():
        from flask import g
        from ..services.encryption import encrypt_pii
        data = request.get_json()
        db = get_db()
        updates = []
        params = []
        if "full_name" in data:
            updates.append("full_name = ?")
            params.append(encrypt_pii(data["full_name"].strip()))
        if "email" in data:
            updates.append("email = ?")
            params.append(encrypt_pii(data["email"].strip().lower()))
        if updates:
            params.append(g.current_user["id"])
            db.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
            db.commit()
        from ..models.user import get_user_by_id, user_to_dict
        user = get_user_by_id(g.current_user["id"])
        return jsonify({"user": user_to_dict(user)}), 200
    return _inner()
