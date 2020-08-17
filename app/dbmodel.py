# Database model

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app import db, login


class User(UserMixin, db.Model):
    """Data model for user accounts."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Person(db.Model):
    """Data model for user information."""
    person_id = db.Column(db.Integer, primary_key=True)
    dispname = db.Column(db.String(64), index=True, unique=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(64))

    def __repr__(self):
        return '<Person {}>'.format(self.dispname)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))
