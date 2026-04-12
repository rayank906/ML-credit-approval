"""Seed the database with demo users and sample applications."""
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app import create_app


SAMPLE_APPLICANTS = [
    {"email": "alice@example.com", "password": "password123", "full_name": "Alice Johnson"},
    {"email": "bob@example.com", "password": "password123", "full_name": "Bob Smith"},
    {"email": "carol@example.com", "password": "password123", "full_name": "Carol Williams"},
]

OFFICER = {"email": "officer@bank.com", "password": "officer123", "full_name": "Officer Davis", "role": "officer"}
ADMIN = {"email": "admin@bank.com", "password": "admin123", "full_name": "Admin Thompson", "role": "admin"}

SAMPLE_APPLICATIONS = [
    {"gender": 1, "own_car": 1, "own_realty": 1, "children": 0, "income": 150000, "income_type": 0,
     "education": 3, "family_status": 1, "housing": 4, "occupation": 3, "work_phone": 1,
     "phone": 1, "email": 1, "family_members": 2, "age": 42, "years_employed": 12},
    {"gender": 0, "own_car": 0, "own_realty": 0, "children": 3, "income": 28000, "income_type": 0,
     "education": 1, "family_status": 0, "housing": 1, "occupation": 0, "work_phone": 0,
     "phone": 0, "email": 0, "family_members": 4, "age": 23, "years_employed": 1},
    {"gender": 1, "own_car": 1, "own_realty": 0, "children": 1, "income": 75000, "income_type": 1,
     "education": 2, "family_status": 3, "housing": 3, "occupation": 5, "work_phone": 1,
     "phone": 0, "email": 1, "family_members": 3, "age": 35, "years_employed": 5},
    {"gender": 0, "own_car": 1, "own_realty": 1, "children": 2, "income": 200000, "income_type": 3,
     "education": 4, "family_status": 1, "housing": 4, "occupation": 17, "work_phone": 1,
     "phone": 1, "email": 1, "family_members": 4, "age": 50, "years_employed": 20},
    {"gender": 1, "own_car": 0, "own_realty": 0, "children": 0, "income": 18000, "income_type": 4,
     "education": 2, "family_status": 0, "housing": 0, "occupation": 18, "work_phone": 0,
     "phone": 1, "email": 1, "family_members": 1, "age": 21, "years_employed": 0.5},
    {"gender": 0, "own_car": 0, "own_realty": 1, "children": 1, "income": 95000, "income_type": 0,
     "education": 3, "family_status": 1, "housing": 4, "occupation": 6, "work_phone": 1,
     "phone": 1, "email": 1, "family_members": 3, "age": 38, "years_employed": 8},
    {"gender": 1, "own_car": 1, "own_realty": 1, "children": 0, "income": 120000, "income_type": 1,
     "education": 4, "family_status": 3, "housing": 4, "occupation": 5, "work_phone": 1,
     "phone": 1, "email": 1, "family_members": 2, "age": 45, "years_employed": 15},
    {"gender": 0, "own_car": 0, "own_realty": 0, "children": 4, "income": 35000, "income_type": 0,
     "education": 0, "family_status": 2, "housing": 2, "occupation": 12, "work_phone": 0,
     "phone": 0, "email": 0, "family_members": 5, "age": 30, "years_employed": 2},
]


def seed():
    app = create_app()

    with app.app_context():
        from app.models.user import create_user, get_user_by_email
        from app.models.application import create_application
        from app.models.audit_log import log_event
        from app.models.notification import create_notification
        from app.extensions import get_db

        # Create officer and admin
        if not get_user_by_email(OFFICER["email"]):
            create_user(**OFFICER)
            print(f"Created officer: {OFFICER['email']}")
        if not get_user_by_email(ADMIN["email"]):
            create_user(**ADMIN)
            print(f"Created admin: {ADMIN['email']}")

        ml_engine = app.config["ML_ENGINE"]

        for i, applicant_data in enumerate(SAMPLE_APPLICANTS):
            user = get_user_by_email(applicant_data["email"])
            if not user:
                user = create_user(**applicant_data)
                print(f"Created applicant: {applicant_data['email']}")

            # Submit 2-3 applications per applicant
            apps_for_user = SAMPLE_APPLICATIONS[i * 2: i * 2 + 3]
            for app_data in apps_for_user:
                ml_results = ml_engine.predict(app_data)
                app_record = create_application(user["id"], app_data, ml_results)

                log_event("application_submitted", application_id=app_record["id"], user_id=user["id"],
                          details={"status": "ml_reviewed"})
                log_event("ml_scored", application_id=app_record["id"], user_id=user["id"],
                          details={"decision": ml_results["decision"], "risk": ml_results["probability_risky"]})

                create_notification(user["id"], "application_update", "Application Submitted",
                                    f"Application #{app_record['id']} scored: {ml_results['decision']}",
                                    f"/app/applications/{app_record['id']}")

                print(f"  Created application #{app_record['id']} for {applicant_data['email']} "
                      f"-> {ml_results['decision']} (risk: {ml_results['probability_risky']}%)")

        # Officer reviews some applications
        db = get_db()
        officer = get_user_by_email(OFFICER["email"])
        pending = db.execute("SELECT id, ml_decision FROM applications WHERE status = 'ml_reviewed' LIMIT 4").fetchall()
        for app_row in pending:
            decision = dict(app_row)["ml_decision"]
            if random.random() < 0.3:
                decision = "approved" if decision == "rejected" else "rejected"

            db.execute(
                "INSERT INTO officer_reviews (application_id, officer_id, override_decision, justification) VALUES (?, ?, ?, ?)",
                (app_row["id"], officer["id"],
                 decision if decision != dict(app_row)["ml_decision"] else None,
                 "Reviewed applicant profile and supporting documents."),
            )
            db.execute(
                "UPDATE applications SET status = ?, final_decision = ?, updated_at = datetime('now') WHERE id = ?",
                (decision, decision, app_row["id"]),
            )
            db.commit()

            log_event("officer_review", application_id=app_row["id"], user_id=officer["id"],
                      details={"decision": decision})
            print(f"  Officer reviewed application #{app_row['id']} -> {decision}")

        print("\nSeed complete!")
        print(f"  Officer login: {OFFICER['email']} / {OFFICER['password']}")
        print(f"  Admin login: {ADMIN['email']} / {ADMIN['password']}")
        for a in SAMPLE_APPLICANTS:
            print(f"  Applicant login: {a['email']} / {a['password']}")


if __name__ == "__main__":
    seed()
