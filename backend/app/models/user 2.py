import bcrypt

from ..extensions import get_db
from ..services.encryption import encrypt_pii, decrypt_pii


def create_user(email, password, full_name, role="applicant"):
    db = get_db()
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    encrypted_email = encrypt_pii(email)
    encrypted_name = encrypt_pii(full_name)

    cursor = db.execute(
        "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
        (encrypted_email, password_hash, encrypted_name, role),
    )
    db.commit()
    return get_user_by_id(cursor.lastrowid)


def get_user_by_id(user_id):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,)).fetchone()
    if row:
        return _decrypt_user(dict(row))
    return None


def get_user_by_email(email):
    db = get_db()
    rows = db.execute("SELECT * FROM users WHERE is_active = 1").fetchall()
    for row in rows:
        user = dict(row)
        decrypted_email = decrypt_pii(user["email"])
        if decrypted_email == email:
            return _decrypt_user(user)
    return None


def verify_password(stored_hash, password):
    return bcrypt.checkpw(password.encode(), stored_hash.encode())


def _decrypt_user(user):
    user["email"] = decrypt_pii(user["email"])
    user["full_name"] = decrypt_pii(user["full_name"])
    user.pop("password_hash", None)
    return user


def user_to_dict(user):
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }
