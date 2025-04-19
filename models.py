from app import db
from datetime import datetime

class CheckResult(db.Model):
    """Model to store URL check results"""
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(2048), nullable=False)
    dt_prediction = db.Column(db.Integer)  # Decision Tree prediction (0 or 1)
    rf_prediction = db.Column(db.Integer)  # Random Forest prediction (0 or 1)
    xgb_prediction = db.Column(db.Integer)  # XGBoost prediction (0 or 1)
    dt_confidence = db.Column(db.Float)  # Decision Tree confidence score
    rf_confidence = db.Column(db.Float)  # Random Forest confidence score
    xgb_confidence = db.Column(db.Float)  # XGBoost confidence score
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<CheckResult {self.url}>'
