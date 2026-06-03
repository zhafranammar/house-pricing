"""Flask application factory."""

import os

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("SECRET_KEY", "dev-uas-data-mining")

    from routes.dataset import bp as dataset_bp
    from routes.main import bp as main_bp
    from routes.predict import bp as predict_bp
    from routes.tasks import bp as tasks_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(predict_bp)

    return app
