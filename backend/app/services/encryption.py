from cryptography.fernet import Fernet
from flask import current_app

_fernet = None


def _get_fernet():
    global _fernet
    if _fernet is None:
        key = current_app.config["ENCRYPTION_KEY"].encode()
        _fernet = Fernet(key)
    return _fernet


def encrypt_pii(plaintext):
    if not plaintext:
        return plaintext
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_pii(ciphertext):
    if not ciphertext:
        return ciphertext
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except Exception:
        return ciphertext
