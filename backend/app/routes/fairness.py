from flask import Blueprint, jsonify, request

from ..middleware.auth_middleware import require_role
from ..services.fairness_audit import run_audit, DEMOGRAPHIC_COLUMNS

fairness_bp = Blueprint("fairness", __name__, url_prefix="/api/fairness")


@fairness_bp.route("/audit", methods=["GET"])
@require_role("officer", "admin")
def audit():
    demographic = request.args.get("demographic", "gender")
    result = run_audit(demographic)
    if result is None:
        return jsonify({"error": f"Invalid demographic. Choose from: {list(DEMOGRAPHIC_COLUMNS.keys())}"}), 400
    return jsonify(result), 200
