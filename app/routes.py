import tensorflow as tf
import numpy as np
import os
import pickle
import shutil
import cv2

from flask import render_template, Response, redirect, flash, url_for, request
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
from werkzeug.utils import secure_filename
from importlib import import_module

import forms
from app import app, db
from app.dbmodel import User, Person
from camera_opencv import Camera
from models import Models, detect_faces
from utils import generate_bbox
from forms import validate_dispname

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
# else:
#     from camera import Camera

model = Models(model='senet50')


@app.route('/')
@app.route('/index')
@login_required
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = forms.LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('auth/login.html', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = forms.FaceRegForm()
    query = Person.query.all()
    if form.validate_on_submit():
        if request.method == 'POST':
            if validate_dispname(form.dispname.data) == 0:
                flash('Please use a different display name.')
                return redirect(request.url)
            # if 'image' not in request.files:
            #     return redirect('No file part')
            file = request.files['image']
            # if file.filename == '':
            #     flash('No selected file')
            if file and allowed_file(file.filename):
                # upload file to directory
                filename = secure_filename(file.filename)
                folder_name = str(form.dispname.data).replace(' ', '_')
                os.makedirs(os.getcwd() + app.config['UPLOAD_FOLDER'] + '\\' + folder_name, exist_ok=True)
                savepath = os.path.join(os.getcwd() + app.config['UPLOAD_FOLDER'] + '\\' + folder_name, filename)
                file.save(savepath)
                # detect face(s) in uploaded image
                img = cv2.imread(savepath)
                face_boxes = detect_faces(img, 'mtcnn', scale=1/4)
                # add embedding(s) to .dat file
                with open('embeddings/senet50_label_emb.dat', 'rb') as f:
                    _label_emb = pickle.load(f)
                for (x, y, w, h) in face_boxes:
                    r, nx, ny, nr = generate_bbox(x, y, w, h)
                    face = img[ny:ny + nr, nx:nx + nr]
                    face_resize = cv2.resize(face, (224, 224))
                    # generate embedding
                    embedding = model.get_embedding(face_resize)
                    # add
                    _label_emb[form.dispname.data] = embedding
                with open('embeddings/senet50_label_emb.dat', 'wb') as f:
                    pickle.dump(_label_emb, f)
                #  add user information to database
                personToAdd = Person(dispname=form.dispname.data, age=form.age.data, gender=form.gender.data)
                db.session.add(personToAdd)
                db.session.commit()
                return redirect(url_for('register'))
    return render_template('CRUD/face_reg.html', form=form, query=query)


@app.route('/register/<dispname>/update', methods=['GET', 'POST'])
def update(dispname):
    form = forms.UpdateForm()
    personToUpdate = Person.query.filter_by(dispname=dispname).first()
    if form.validate_on_submit():
        if request.method == 'POST':
            # check if dispname is unique
            if validate_dispname(form.new_dispname.data) == 0:
                if form.new_dispname.data != personToUpdate.dispname:
                    flash('Please use a different display name.')
                    return redirect(request.url)
            # update information to database
            personToUpdate.dispname = form.new_dispname.data
            personToUpdate.age = form.new_age.data
            personToUpdate.gender = form.new_gender.data
            db.session.commit()
            # update display name in .dat file
            with open('embeddings/senet50_label_emb.dat', 'rb') as f:
                _label_emb = pickle.load(f)
            _label_emb[str(form.new_dispname.data)] = _label_emb.pop(str(dispname))
            with open('embeddings/senet50_label_emb.dat', 'wb') as f:
                pickle.dump(_label_emb, f)
            # rename directory storing images
            src = 'app/static/' + str(dispname).replace(' ', '_')
            dst = 'app/static/' + str(form.new_dispname.data).replace(' ', '_')
            os.rename(src, dst)
            return redirect(url_for('register'))
    return render_template('CRUD/info_update.html', personToUpdate=personToUpdate, form=form)


@app.route('/register/<dispname>/delete', methods=['GET', 'POST'])
def delete(dispname):
    form = forms.DeleteForm()
    personToDelete = Person.query.filter_by(dispname=dispname).first()
    if form.validate_on_submit():
        if request.method == 'POST':
            # delete information in database
            db.session.delete(personToDelete)
            db.session.commit()
            # delete face embedding(s) in .dat file
            with open('embeddings/senet50_label_emb.dat', 'rb') as f:
                _label_emb = pickle.load(f)
            _label_emb.pop(str(dispname))
            with open('embeddings/senet50_label_emb.dat', 'wb') as f:
                pickle.dump(_label_emb, f)
            # delete image(s) in directory (folder)
            dir_path = 'app/static/' + str(dispname).replace(' ', '_')
            shutil.rmtree(dir_path)
            return redirect(url_for('register'))
    return render_template('CRUD/user_delete.html', personToDelete=personToDelete, form=form)


@app.route('/register/<dispname>/images')
def show_images(dispname):
    personToShow = Person.query.filter_by(dispname=dispname).first()
    pics_dir = os.getcwd() + app.config['UPLOAD_FOLDER'] + '\\' + str(dispname).replace(' ', '_')
    pics_dir = pics_dir.replace('\\', '/')
    pics = os.listdir(pics_dir)
    return render_template('CRUD/show_images.html', personToShow=personToShow, folder=str(dispname).replace(' ', '_'), pics=pics)


def gen(self):
    """Video streaming generator function."""
    while True:
        frame = self.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera')
def camera():
    return render_template('camera.html')


if __name__ == '__main__':
    app.run(host='127.0.0.5', threaded=True, use_reloader=True)
