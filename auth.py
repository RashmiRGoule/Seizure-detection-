"""
auth.py
-------
Handles signup, login, logout using Flask-Login.

Security notes:
  - Passwords are never stored in plain text - werkzeug's
    generate_password_hash (PBKDF2) is used via the User model.
  - This is a project/demo-grade auth flow (good for a college project /
    portfolio demo). For real deployment you would add: email verification,
    rate-limiting on login attempts, HTTPS enforcement, password reset flow,
    and CSRF protection (Flask-WTF) on forms.
"""

import re
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from extensions import db
from models import User

auth_bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("predict.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not EMAIL_RE.match(email):
            flash("Please enter a valid email address.", "error")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("signup.html")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists. Please log in.", "error")
            return redirect(url_for("auth.login"))

        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("signup.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("predict.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if user is None or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        login_user(user)
        flash("Logged in successfully.", "success")
        return redirect(url_for("predict.dashboard"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
