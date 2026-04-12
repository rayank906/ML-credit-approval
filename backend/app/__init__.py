from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .config import Config
from .extensions import close_db, init_db


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[app.config["RATE_LIMIT_DEFAULT"]],
        storage_uri="memory://",
    )

    app.teardown_appcontext(close_db)

    with app.app_context():
        init_db(app)

        from .services.ml_engine import MLEngine
        app.config["ML_ENGINE"] = MLEngine(app.config["MODEL_PATH"])

    from .routes.auth import auth_bp
    from .routes.applications import applications_bp
    from .routes.officer import officer_bp
    from .routes.analytics import analytics_bp
    from .routes.fairness import fairness_bp
    from .routes.simulator import simulator_bp
    from .routes.notifications import notifications_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(applications_bp)
    app.register_blueprint(officer_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(fairness_bp)
    app.register_blueprint(simulator_bp)
    app.register_blueprint(notifications_bp)

    limiter.limit(app.config["RATE_LIMIT_LOGIN"])(auth_bp)

    @app.route("/api/health")
    def health():
        return {"status": "ok"}, 200

    @app.route("/api/mappings")
    def mappings():
        from .services.ml_engine import ALL_MAPPINGS
        return ALL_MAPPINGS, 200

    return app
