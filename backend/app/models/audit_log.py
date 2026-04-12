import json

from ..extensions import get_db


def log_event(action, application_id=None, user_id=None, details=None, ip_address=None):
    db = get_db()
    db.execute(
        "INSERT INTO audit_logs (application_id, user_id, action, details_json, ip_address) VALUES (?, ?, ?, ?, ?)",
        (application_id, user_id, action, json.dumps(details) if details else None, ip_address),
    )
    db.commit()


def get_timeline(application_id):
    db = get_db()
    rows = db.execute(
        """SELECT al.*, u.full_name as user_name
           FROM audit_logs al
           LEFT JOIN users u ON al.user_id = u.id
           WHERE al.application_id = ?
           ORDER BY al.created_at ASC""",
        (application_id,),
    ).fetchall()

    events = []
    for row in rows:
        d = dict(row)
        if d.get("details_json"):
            try:
                d["details"] = json.loads(d["details_json"])
            except (json.JSONDecodeError, TypeError):
                d["details"] = {}
        else:
            d["details"] = {}
        events.append(d)
    return events
