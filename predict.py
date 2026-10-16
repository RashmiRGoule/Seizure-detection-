"""
predict.py
----------
Routes for the logged-in part of the app:
  /            - dashboard with upload form
  /predict     - handles the uploaded EEG CSV, runs inference, shows result
  /history     - lists this user's past predictions

Expected upload format:
  A CSV file containing EEG signal values for ONE report/segment.
  Accepted shapes:
    - A single row of 178 numeric values (with or without a header row)
    - The full UCI-style row format: first column an ID/name, then X1..X178,
      optionally a trailing y/label column (which is ignored if present)
  If the uploaded file has multiple rows, only the FIRST row is used for
  prediction (clearly explained to the user in the UI), since this app is
  built around "one EEG report = one prediction".
"""

import os
import csv
import uuid

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from extensions import db
from models import PredictionHistory
from inference import predict_eeg, get_model_status

predict_bp = Blueprint("predict", __name__)

ALLOWED_EXTENSIONS = {"csv"}
EXPECTED_FEATURE_COUNT = 178


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_eeg_csv(filepath):
    """
    Reads the uploaded CSV and extracts a single row of EEG_FEATURE_COUNT
    numeric values, robust to:
      - a leading ID/name column
      - a trailing label column (y)
      - an optional header row
    Returns a list of floats of length EXPECTED_FEATURE_COUNT.
    Raises ValueError with a user-friendly message if parsing fails.
    """
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError("The uploaded CSV file is empty.")

    # Skip a header row if the first row's first numeric-looking check fails
    def row_numeric_values(row):
        vals = []
        for cell in row:
            try:
                vals.append(float(cell))
            except ValueError:
                continue
        return vals

    data_row = None
    for row in rows:
        nums = row_numeric_values(row)
        if len(nums) >= EXPECTED_FEATURE_COUNT:
            data_row = nums
            break

    if data_row is None:
        raise ValueError(
            f"Could not find a row with at least {EXPECTED_FEATURE_COUNT} numeric "
            f"EEG values. Please upload a CSV matching the expected EEG report format."
        )

    # If there are more than EXPECTED_FEATURE_COUNT numeric values (e.g. a
    # trailing label column got parsed as numeric too), take the first
    # EXPECTED_FEATURE_COUNT values.
    return data_row[:EXPECTED_FEATURE_COUNT]


@predict_bp.route("/")
@login_required
def dashboard():
    model_error = get_model_status()
    return render_template("dashboard.html", model_error=model_error)


@predict_bp.route("/predict", methods=["POST"])
@login_required
def predict():
    model_error = get_model_status()
    if model_error:
        flash(model_error, "error")
        return redirect(url_for("predict.dashboard"))

    if "eeg_file" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("predict.dashboard"))

    file = request.files["eeg_file"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("predict.dashboard"))

    if not allowed_file(file.filename):
        flash("Only .csv files are supported.", "error")
        return redirect(url_for("predict.dashboard"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        values = parse_eeg_csv(filepath)
        result = predict_eeg(values)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("predict.dashboard"))
    except Exception as e:
        flash(f"Unexpected error while processing the file: {e}", "error")
        return redirect(url_for("predict.dashboard"))

    history_entry = PredictionHistory(
        user_id=current_user.id,
        filename=filename,
        result_label=result["label"],
        seizure_probability=result["probability"],
    )
    db.session.add(history_entry)
    db.session.commit()

    return render_template(
        "result.html",
        filename=filename,
        label=result["label"],
        probability=round(result["probability"] * 100, 2),
    )


@predict_bp.route("/history")
@login_required
def history():
    records = (
        PredictionHistory.query.filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )
    return render_template("history.html", records=records)
