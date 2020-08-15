#   forms.py
import os
from flask import flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, IntegerField, SelectField
from wtforms.validators import DataRequired, ValidationError
from app.dbmodel import Person


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()], render_kw={"placeholder": "Username"})
    password = PasswordField('Password', validators=[DataRequired()], render_kw={"placeholder": "Password"})
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


def validate_dispname(dispname):
    display_name = Person.query.filter_by(dispname=dispname).first()
    if display_name is not None:
        return 0
    return 1


class FaceRegForm(FlaskForm):
    dispname = StringField('Display name', validators=[DataRequired()], render_kw={"placeholder": "Name"})
    age = IntegerField('Age', validators=[DataRequired()], render_kw={"placeholder": "Age"})
    gender = SelectField('Gender', choices=[('', 'Select your gender'), ('Male', 'Male'), ('Female', 'Female')],
                         validators=[DataRequired()], coerce=str, render_kw={"placeholder": "Gender"})
    image = FileField('image', validators=[FileRequired()])
    submit = SubmitField('Register')


class UpdateForm(FlaskForm):
    new_dispname = StringField('New display name', validators=[DataRequired()], render_kw={"placeholder": 'New display name'})
    new_age = IntegerField('Age', validators=[DataRequired()], render_kw={"placeholder": 'New age'})
    new_gender = SelectField('Gender', choices=[('', 'Select your gender'), ('Male', 'Male'), ('Female', 'Female')],
                             validators=[DataRequired()], coerce=str, render_kw={"placeholder": "New gender"})
    submit = SubmitField('Update')


class DeleteForm(FlaskForm):
    submit = SubmitField('Delete')
