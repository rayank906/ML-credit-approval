#!/usr/bin/env python3
"""Create a staff (officer / admin) user account from the command line.

Usage:
    python3 seed_staff.py --email admin@bank.com --password s3cret --role admin --name "Bank Admin"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from auth import hash_password  # noqa: E402
from database import SessionLocal, User, init_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a staff user account")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--role", choices=["officer", "admin"], default="admin")
    parser.add_argument("--name", default="Staff User")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == args.email).first()
        if existing:
            existing.role = args.role
            existing.password_hash = hash_password(args.password)
            existing.full_name = args.name
            db.commit()
            print(f"Updated existing user {args.email} -> role={args.role}")
        else:
            user = User(
                email=args.email,
                password_hash=hash_password(args.password),
                full_name=args.name,
                role=args.role,
            )
            db.add(user)
            db.commit()
            print(f"Created {args.role} user: {args.email}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
