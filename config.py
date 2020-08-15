import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    # General Flask Config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'this-is-my-website'

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'sqlite:///' + os.path.join(basedir, 'app.db')
    # SQLALCHEMY_ECHO = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
