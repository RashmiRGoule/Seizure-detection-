"""
models.py
---------
SQLAlchemy models:
  - User: email + hashed password, used for login.
  - PredictionHistory: stores past predictions made by a user, so they can
    review previous EEG report results.
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship("PredictionHistory", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    result_label = db.Column(db.String(50), nullable=False)       # "Seizure" / "Non-Seizure"
    seizure_probability = db.Column(db.Float, nullable=False)     # 0.0 - 1.0
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
