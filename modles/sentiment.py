from datetime import datetime
from . import db

class Sentiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('sentiments', lazy=True))

    def __repr__(self):
        return f"<Sentiment {self.id}, Prediction {self.prediction}>"
