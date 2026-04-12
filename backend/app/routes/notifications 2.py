from flask import Blueprint, g, jsonify, request

from ..middleware.auth_middleware import require_auth
from ..models.notification import get_notifications, mark_read, mark_all_read

notifications_bp = Blueprint("notifications", __name__, url_prefix="/api/notifications")


@notifications_bp.route("", methods=["GET"])
@require_auth
def list_notifications():
    unread_only = request.args.get("unread_only", "false").lower() == "true"
    result = get_notifications(g.current_user["id"], unread_only=unread_only)
    return jsonify(result), 200


@notifications_bp.route("/<int:notif_id>/read", methods=["PUT"])
@require_auth
def read_notification(notif_id):
    mark_read(notif_id, g.current_user["id"])
    return jsonify({"success": True}), 200


@notifications_bp.route("/read-all", methods=["PUT"])
@require_auth
def read_all_notifications():
    count = mark_all_read(g.current_user["id"])
    return jsonify({"updated_count": count}), 200
