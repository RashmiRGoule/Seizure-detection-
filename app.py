"""
app.py
------
Flask web app for early seizure detection from EEG data.

Pages:
  /signup   - create account (email + password)
  /login    - login
  /logout   - logout
  /         - dashboard: upload EEG CSV report -> get prediction
  /history  - past predictions for the logged-in user

Run:
  python app.py
Then open http://127.0.0.1:5000
"""

import os
from flask import Flask
from flask_login import LoginManager

from extensions import db
from models import User


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-this-in-production")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "instance", "app.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance"), exist_ok=True)

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "info"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from auth import auth_bp
    from predict import predict_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)

    with app.app_context():
        db.create_all()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
