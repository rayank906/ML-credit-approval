from flask import Blueprint, g, jsonify, request

from ..middleware.auth_middleware import require_role
from ..models.application import get_application, get_all_applications, update_application, get_applications_for_user
from ..models.audit_log import log_event
from ..models.notification import create_notification
from ..extensions import get_db

officer_bp = Blueprint("officer", __name__, url_prefix="/api/officer")


@officer_bp.route("/queue", methods=["GET"])
@require_role("officer", "admin")
def review_queue():
    status = request.args.get("status", "ml_reviewed")
    sort = request.args.get("sort", "risk_desc")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))

    result = get_all_applications(status=status, sort=sort, page=page, per_page=per_page)
    return jsonify(result), 200


@officer_bp.route("/review/<int:app_id>", methods=["POST"])
@require_role("officer", "admin")
def review_application(app_id):
    app_record = get_application(app_id)
    if not app_record:
        return jsonify({"error": "Application not found"}), 404

    data = request.get_json()
    decision = data.get("decision")
    justification = data.get("justification", "").strip()

    if decision not in ("approved", "rejected"):
        return jsonify({"error": "Decision must be 'approved' or 'rejected'"}), 400
    if not justification:
        return jsonify({"error": "Justification is required"}), 400

    db = get_db()

    is_override = decision != app_record.get("ml_decision")
    db.execute(
        "INSERT INTO officer_reviews (application_id, officer_id, override_decision, justification) VALUES (?, ?, ?, ?)",
        (app_id, g.current_user["id"], decision if is_override else None, justification),
    )

    update_application(app_id, status=decision, final_decision=decision)

    log_event(
        "officer_review",
        application_id=app_id,
        user_id=g.current_user["id"],
        details={
            "decision": decision,
            "is_override": is_override,
            "ml_decision": app_record.get("ml_decision"),
            "justification": justification,
        },
        ip_address=request.remote_addr,
    )

    create_notification(
        app_record["user_id"],
        "decision_made",
        f"Application #{app_id} — {decision.title()}",
        f"A bank officer has reviewed your application and it has been {decision}.",
        f"/app/applications/{app_id}",
    )

    updated = get_application(app_id)
    review = db.execute(
        "SELECT * FROM officer_reviews WHERE application_id = ? ORDER BY reviewed_at DESC LIMIT 1",
        (app_id,),
    ).fetchone()

    return jsonify({
        "application": updated,
        "review": dict(review) if review else None,
    }), 200


@officer_bp.route("/customers/<int:user_id>", methods=["GET"])
@require_role("officer", "admin")
def customer_profile(user_id):
    from ..models.user import get_user_by_id, user_to_dict
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "Customer not found"}), 404

    apps = get_applications_for_user(user_id, page=1, per_page=100)

    risk_trajectory = []
    for app_rec in apps["applications"]:
        risk_trajectory.append({
            "date": app_rec["created_at"],
            "risk_score": app_rec.get("blended_risk_score"),
            "credit_score": app_rec.get("credit_score"),
            "decision": app_rec.get("final_decision") or app_rec.get("ml_decision"),
        })

    return jsonify({
        "customer": user_to_dict(user),
        "applications": apps["applications"],
        "risk_trajectory": risk_trajectory,
        "total_applications": apps["total"],
    }), 200
