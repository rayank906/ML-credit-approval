from ..extensions import get_db


def create_notification(user_id, notif_type, title, message, link=None):
    db = get_db()
    db.execute(
        "INSERT INTO notifications (user_id, type, title, message, link) VALUES (?, ?, ?, ?, ?)",
        (user_id, notif_type, title, message, link),
    )
    db.commit()


def get_notifications(user_id, unread_only=False, limit=50):
    db = get_db()
    query = "SELECT * FROM notifications WHERE user_id = ?"
    params = [user_id]
    if unread_only:
        query += " AND is_read = 0"
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(query, params).fetchall()
    unread_count = db.execute(
        "SELECT COUNT(*) FROM notifications WHERE user_id = ? AND is_read = 0", (user_id,)
    ).fetchone()[0]

    return {
        "notifications": [dict(r) for r in rows],
        "unread_count": unread_count,
    }


def mark_read(notif_id, user_id):
    db = get_db()
    db.execute("UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?", (notif_id, user_id))
    db.commit()


def mark_all_read(user_id):
    db = get_db()
    result = db.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ? AND is_read = 0", (user_id,))
    db.commit()
    return result.rowcount
