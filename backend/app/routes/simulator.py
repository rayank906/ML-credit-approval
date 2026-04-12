import json

from flask import Blueprint, current_app, g, jsonify, request

from ..middleware.auth_middleware import require_auth
from ..services.validation import validate_application_input
from ..extensions import get_db

simulator_bp = Blueprint("simulator", __name__, url_prefix="/api/simulator")


@simulator_bp.route("/what-if", methods=["POST"])
@require_auth
def what_if():
    data = request.get_json()
    cleaned, errors = validate_application_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    ml_engine = current_app.config["ML_ENGINE"]
    result = ml_engine.predict(cleaned)

    db = get_db()
    db.execute(
        "INSERT INTO what_if_simulations (user_id, input_json, result_json) VALUES (?, ?, ?)",
        (g.current_user["id"], json.dumps(cleaned), json.dumps(result)),
    )
    db.commit()

    return jsonify(result), 200


@simulator_bp.route("/what-if/compare", methods=["POST"])
@require_auth
def what_if_compare():
    data = request.get_json()
    base_data = data.get("base")
    modified_data = data.get("modified")

    if not base_data or not modified_data:
        return jsonify({"error": "Both 'base' and 'modified' feature sets are required"}), 400

    base_cleaned, base_errors = validate_application_input(base_data)
    if base_errors:
        return jsonify({"error": "Base validation failed", "details": base_errors}), 400

    mod_cleaned, mod_errors = validate_application_input(modified_data)
    if mod_errors:
        return jsonify({"error": "Modified validation failed", "details": mod_errors}), 400

    ml_engine = current_app.config["ML_ENGINE"]
    base_result = ml_engine.predict(base_cleaned)
    mod_result = ml_engine.predict(mod_cleaned)

    diff = {
        "risk_change": round(mod_result["probability_risky"] - base_result["probability_risky"], 1),
        "credit_score_change": mod_result["credit_score"] - base_result["credit_score"],
        "credit_limit_change": mod_result["recommended_credit_limit"] - base_result["recommended_credit_limit"],
        "decision_changed": base_result["decision"] != mod_result["decision"],
    }

    return jsonify({
        "base": base_result,
        "modified": mod_result,
        "diff": diff,
    }), 200
