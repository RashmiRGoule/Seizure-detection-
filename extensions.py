"""
extensions.py
-------------
Holds the SQLAlchemy db object separately so app.py, models.py, auth.py,
and predict.py can all import it without circular-import problems.
"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
