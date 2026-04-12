from flask import Blueprint, current_app, g, jsonify, request

from ..middleware.auth_middleware import require_auth
from ..models.application import (
    create_application, get_application, get_applications_for_user, get_all_applications,
)
from ..models.audit_log import log_event, get_timeline
from ..models.notification import create_notification
from ..services.validation import validate_application_input

applications_bp = Blueprint("applications", __name__, url_prefix="/api/applications")


@applications_bp.route("", methods=["POST"])
@require_auth
def submit_application():
    data = request.get_json()
    cleaned, errors = validate_application_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    ml_engine = current_app.config["ML_ENGINE"]
    ml_results = ml_engine.predict(cleaned)

    app_record = create_application(g.current_user["id"], cleaned, ml_results)

    log_event(
        "application_submitted",
        application_id=app_record["id"],
        user_id=g.current_user["id"],
        details={"status": "ml_reviewed", "ml_decision": ml_results["decision"]},
        ip_address=request.remote_addr,
    )
    log_event(
        "ml_scored",
        application_id=app_record["id"],
        user_id=g.current_user["id"],
        details={
            "credit_score": ml_results["credit_score"],
            "risk_score": ml_results["probability_risky"],
            "decision": ml_results["decision"],
            "model_version": ml_results["model_version"],
        },
    )

    create_notification(
        g.current_user["id"],
        "application_update",
        "Application Submitted",
        f"Your application #{app_record['id']} has been submitted and scored. Decision: {ml_results['decision'].title()}",
        f"/app/applications/{app_record['id']}",
    )

    return jsonify({"application": app_record, "ml_results": ml_results}), 201


@applications_bp.route("", methods=["GET"])
@require_auth
def list_applications():
    status = request.args.get("status")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))

    if g.current_user["role"] in ("officer", "admin"):
        sort = request.args.get("sort", "created_at_desc")
        result = get_all_applications(status=status, sort=sort, page=page, per_page=per_page)
    else:
        result = get_applications_for_user(g.current_user["id"], status=status, page=page, per_page=per_page)

    return jsonify(result), 200


@applications_bp.route("/<int:app_id>", methods=["GET"])
@require_auth
def get_application_detail(app_id):
    app_record = get_application(app_id)
    if not app_record:
        return jsonify({"error": "Application not found"}), 404

    if g.current_user["role"] == "applicant" and app_record["user_id"] != g.current_user["id"]:
        return jsonify({"error": "Access denied"}), 403

    from ..extensions import get_db
    db = get_db()
    reviews = db.execute(
        """SELECT r.*, u.full_name as officer_name
           FROM officer_reviews r
           LEFT JOIN users u ON r.officer_id = u.id
           WHERE r.application_id = ?
           ORDER BY r.reviewed_at DESC""",
        (app_id,),
    ).fetchall()

    return jsonify({
        "application": app_record,
        "reviews": [dict(r) for r in reviews],
    }), 200


@applications_bp.route("/<int:app_id>/timeline", methods=["GET"])
@require_auth
def get_application_timeline(app_id):
    app_record = get_application(app_id)
    if not app_record:
        return jsonify({"error": "Application not found"}), 404

    if g.current_user["role"] == "applicant" and app_record["user_id"] != g.current_user["id"]:
        return jsonify({"error": "Access denied"}), 403

    events = get_timeline(app_id)
    return jsonify({"events": events}), 200
