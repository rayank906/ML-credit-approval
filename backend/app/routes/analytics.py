from flask import Blueprint, jsonify, request

from ..middleware.auth_middleware import require_role
from ..extensions import get_db

analytics_bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


@analytics_bp.route("/summary", methods=["GET"])
@require_role("officer", "admin")
def summary():
    db = get_db()
    period = request.args.get("period", "30")
    try:
        days = int(period)
    except ValueError:
        days = 30

    row = db.execute(f"""
        SELECT
            COUNT(*) as total_apps,
            SUM(CASE WHEN final_decision = 'approved' OR (final_decision IS NULL AND ml_decision = 'approved') THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN final_decision = 'rejected' OR (final_decision IS NULL AND ml_decision = 'rejected') THEN 1 ELSE 0 END) as rejected,
            AVG(blended_risk_score) as avg_risk,
            AVG(credit_score) as avg_credit_score,
            AVG(recommended_credit_limit) as avg_credit_limit
        FROM applications
        WHERE created_at >= datetime('now', '-{days} days')
    """).fetchone()

    total = row["total_apps"] or 0
    approved = row["approved"] or 0

    return jsonify({
        "total_applications": total,
        "approved": approved,
        "rejected": row["rejected"] or 0,
        "pending": total - approved - (row["rejected"] or 0),
        "approval_rate": round(approved / total * 100, 1) if total > 0 else 0,
        "avg_risk_score": round((row["avg_risk"] or 0) * 100, 1),
        "avg_credit_score": round(row["avg_credit_score"] or 0),
        "avg_credit_limit": round(row["avg_credit_limit"] or 0, -2),
        "period_days": days,
    }), 200


@analytics_bp.route("/trends", methods=["GET"])
@require_role("officer", "admin")
def trends():
    db = get_db()
    period = int(request.args.get("period", "90"))

    rows = db.execute(f"""
        SELECT
            date(created_at) as date,
            COUNT(*) as total,
            SUM(CASE WHEN final_decision = 'approved' OR (final_decision IS NULL AND ml_decision = 'approved') THEN 1 ELSE 0 END) as approved,
            AVG(blended_risk_score) as avg_risk
        FROM applications
        WHERE created_at >= datetime('now', '-{period} days')
        GROUP BY date(created_at)
        ORDER BY date(created_at)
    """).fetchall()

    return jsonify({
        "trends": [{
            "date": r["date"],
            "total": r["total"],
            "approved": r["approved"],
            "rejected": r["total"] - r["approved"],
            "approval_rate": round(r["approved"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
            "avg_risk": round((r["avg_risk"] or 0) * 100, 1),
        } for r in rows],
    }), 200


@analytics_bp.route("/risk-distribution", methods=["GET"])
@require_role("officer", "admin")
def risk_distribution():
    db = get_db()
    rows = db.execute("""
        SELECT
            CASE
                WHEN blended_risk_score < 0.1 THEN '0-10%'
                WHEN blended_risk_score < 0.2 THEN '10-20%'
                WHEN blended_risk_score < 0.3 THEN '20-30%'
                WHEN blended_risk_score < 0.4 THEN '30-40%'
                WHEN blended_risk_score < 0.5 THEN '40-50%'
                WHEN blended_risk_score < 0.6 THEN '50-60%'
                WHEN blended_risk_score < 0.7 THEN '60-70%'
                WHEN blended_risk_score < 0.8 THEN '70-80%'
                WHEN blended_risk_score < 0.9 THEN '80-90%'
                ELSE '90-100%'
            END as bucket,
            COUNT(*) as count
        FROM applications
        WHERE blended_risk_score IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    """).fetchall()

    return jsonify({
        "distribution": [{"bucket": r["bucket"], "count": r["count"]} for r in rows],
    }), 200


@analytics_bp.route("/decision-breakdown", methods=["GET"])
@require_role("officer", "admin")
def decision_breakdown():
    db = get_db()
    row = db.execute("""
        SELECT
            SUM(CASE WHEN ml_decision = 'approved' AND (final_decision IS NULL OR final_decision = 'approved') THEN 1 ELSE 0 END) as ml_approved,
            SUM(CASE WHEN ml_decision = 'rejected' AND (final_decision IS NULL OR final_decision = 'rejected') THEN 1 ELSE 0 END) as ml_rejected,
            SUM(CASE WHEN ml_decision = 'rejected' AND final_decision = 'approved' THEN 1 ELSE 0 END) as overridden_to_approved,
            SUM(CASE WHEN ml_decision = 'approved' AND final_decision = 'rejected' THEN 1 ELSE 0 END) as overridden_to_rejected,
            COUNT(*) as total
        FROM applications
        WHERE status != 'pending'
    """).fetchone()

    return jsonify({
        "ml_approved": row["ml_approved"] or 0,
        "ml_rejected": row["ml_rejected"] or 0,
        "overridden_to_approved": row["overridden_to_approved"] or 0,
        "overridden_to_rejected": row["overridden_to_rejected"] or 0,
        "total": row["total"] or 0,
    }), 200
