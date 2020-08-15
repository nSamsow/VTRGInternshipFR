import tensorflow as tf
import os
import time
import numpy as np
import cv2
import pickle
from base_camera import BaseCamera
from models import Models, detect_faces
from utils import generate_bbox, find_most_sim_face, recognized_box_n_text, \
                  unrecognized_box_n_text, show_proc_time

model_metric_threshold = {'vgg16': {'cosine': 0.57125255, 'l2_distance': 1.1425071},
                          'resnet50': {'cosine': 0.54815352, 'l2_distance': 1.0963079},
                          'senet50': {'cosine': 0.49045461, 'l2_distance': 0.98090822}}

# Settings
face_detector = 'mtcnn'
model = 'senet50'
metric = 'l2_distance'
threshold = model_metric_threshold[model][metric]

# Load embeddings from database
with open('embeddings/' + str(model) + '_label_emb.dat', 'rb') as f:
    _label_emb = pickle.load(f)
face_names = list(_label_emb.keys())
known_emb = np.array([emb.reshape(-1, ) for emb in _label_emb.values()])

# Import model
model = Models(model=model)


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        proc_time, detection_time, match_time = [0], [0], [0]
        while True:
            time_0 = time.time()
            # read a frame from camera
            ret, raw_img = camera.read()
            if not ret:
                print("Failed to grab frame")
                continue
            # detect face(s) in a frame
            detection_time_0 = time.time()
            face_boxes = detect_faces(raw_img, face_detector, scale=1/4)
            detection_time_1 = time.time()
            detection_time.append(detection_time_1 - detection_time_0)
            # compare face(s) to all embeddings in .dat file
            match_time_0 = time.time()
            if len(face_boxes) == 0:
                pass
            else:
                # (x, y, w, h): determine region of each face
                for (x, y, w, h) in face_boxes:
                    r, nx, ny, nr = generate_bbox(x, y, w, h)
                    face = raw_img[ny:ny + nr, nx:nx + nr]
                    face_resize = cv2.resize(face, (224, 224))
                    emb = model.get_embedding(face_resize).astype('float32')

                    name, score = find_most_sim_face(face_names, known_emb, emb,
                                                     metric=metric, threshold=threshold)

                    if name is None:    # unrecognized
                        unrecognized_box_n_text(raw_img, x, y, w, h)
                    else:   # recognized
                        recognized_box_n_text(raw_img, name, score, x, y, w, h)
            match_time_1 = time.time()
            match_time.append(match_time_1 - match_time_0)
            # display processing time
            show_proc_time(raw_img, proc_time, detection_time, match_time)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', raw_img)[1].tobytes()
            #
            time_1 = time.time()
            proc_time.append(time_1 - time_0)
            if len(proc_time) == 31:
                proc_time.pop(0)
                detection_time.pop(0)
                match_time.pop(0)
