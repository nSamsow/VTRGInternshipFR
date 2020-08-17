import tensorflow as tf
from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

app = Flask(__name__, static_url_path="/static", static_folder="static")
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'

# Image Upload Config
UPLOAD_FOLDER = '\\app\\static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from app import routes, dbmodel
from app.dbmodel import User, Person


@app.shell_context_processor
def make_shell_context():
    return dict(db=db, User=User, Person=Person, PersonImage=PersonImage)
