# -*- coding: utf-8 -*-
import os
import shutil

from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_dropzone import Dropzone

from PIL import Image
import cv2 as cv
from datetime import datetime
import uuid
import torch

# from scripts.utils import transform_image as T
from scripts.utils import load_model_detection, load_model_classification, object_detection_api

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
Bootstrap(app)
app.jinja_env.filters['zip'] = zip
app.jinja_env.add_extension('jinja2.ext.loopcontrols')

app.config.update(
    TEMPLATES_AUTO_RELOAD=True,
    DEFAULT_UPLOADED_PATH=os.path.join(basedir, 'static', 'img', 'default_uploads'),
    UPLOADED_PATH=os.path.join(basedir, 'static', 'img', 'uploads'),
    PREDICTED_PATH=os.path.join(basedir, 'static', 'img', 'predicted'),
    # MODELS_PATH=os.path.join(basedir, 'models'),
    # PREVIOUS_PREDICTED_PATH='',
    MODEL_OD_PATH=os.path.join(basedir, 'models', 'car_detection', 'car_only', 'resnet18_900_8/1'),
    MODEL_CL_PATH=os.path.join(basedir, 'models', 'car_classification', 'resnet34_224_50_grad_cam/1'),
    MODEL_NAME='ResNet34',
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_FILENAME='',
    DROPZONE_REDIRECT_VIEW='main_view'  # set redirect view
)
MODEL_OD, img_size_od = load_model_detection(app.config['MODEL_OD_PATH'])
MODEL_CL, img_size_cl = load_model_classification(app.config['MODEL_CL_PATH'])

dropzone = Dropzone(app)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/img'),
                          'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
# Simple http endpoint
@app.route('/hello')
def hello():

    return "Hello World!"

@app.route('/', methods=['POST', 'GET'])
def main_view():
    default_uploads = os.listdir(os.path.join(app.root_path, 'static', 'img', 'default_uploads'))
    # print(app.config['PREVIOUS_PREDICTED_PATH'])
    # try:
    #     shutil.rmtree(app.config['PREVIOUS_PREDICTED_PATH'])
    # except OSError:
    #     pass

    if request.method == 'GET':
        model_name = request.args.get('model_name')
        filename = request.args.get('filename')

        print(model_name)

        if model_name is not None:
            if model_name == 'ResNet50':
                model_name_path = 'resnet50_224_25_grad_cam/1'
            elif model_name == 'ResNet34':
                model_name_path = 'resnet34_224_50_grad_cam/1'
            elif model_name == 'SE ResNet34':
                model_name_path = 'se_linear_resnet34_224_50_grad_cam/1'
            elif model_name == 'SE ResNet50':
                model_name_path = 'se_resnet50_224_50_grad_cam/1'
            elif model_name == 'CBAM ResNet34':
                model_name_path = 'cbam_linear_resnet34_224_50_grad_cam/1'
            elif model_name == 'CBAM ResNet50':
                model_name_path = 'cbam_resnet50_224_50_grad_cam/1'

            app.config.update(
                MODEL_CL_PATH=os.path.join(basedir, 'models', 'car_classification', model_name_path),
                MODEL_NAME=model_name,
                # Flask-Dropzone config:
                DROPZONE_FILENAME=''
            )

            global MODEL_CL, img_size_cl
            MODEL_CL, img_size_cl = load_model_classification(app.config['MODEL_CL_PATH'])

        if filename is not None:
            src_dir = os.path.join(app.config['DEFAULT_UPLOADED_PATH'], filename)
            dst_dir = os.path.join(app.config['UPLOADED_PATH'], filename)
            shutil.copy(src_dir, dst_dir)
            print('copiez')
            app.config.update(
                # Flask-Dropzone config:
                DROPZONE_FILENAME=filename
            )

    if request.method == 'POST':
        for key, f in request.files.items():
            if key.startswith('file'):
                filename = os.path.join(app.config['UPLOADED_PATH'], f.filename)
                f.save(filename)
                app.config.update(
                    # Flask-Dropzone config:
                    DROPZONE_FILENAME=f.filename
                )

    print('Current filename {}'.format(app.config['DROPZONE_FILENAME']))

    images = []
    keep_class = []
    img_path = ''

    if app.config['DROPZONE_FILENAME'] is not '':
        img_path = os.path.join(app.config['UPLOADED_PATH'], app.config['DROPZONE_FILENAME'])

        # current date and time
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        
        folder_name = str(uuid.uuid4().hex)
        predicted_path = os.path.join(app.config['PREDICTED_PATH'], folder_name)
        os.makedirs(predicted_path)

        images, keep_class = object_detection_api(MODEL_OD, MODEL_CL, img_size_od, img_size_cl, 
                                img_path, predicted_path=predicted_path, threshold=0.6, rect_th=2, text_size=2, text_th=2)

        # os.remove(img_path)

    return render_template('index.html', images=images, classes=keep_class, default_uploads=default_uploads, model_name=app.config['MODEL_NAME'])


# @app.route('/predict', methods=['POST', 'GET'])
# def predict():

#     if request.method == 'GET':
#         model_name = request.args.get('model_name')

#         if model_name is not None:
#         #     if model_name == 'ResNet50':
#         #         model_name = 'resnet50_224_25_grad_cam/1'
#         #     elif model_name == 'ResNet50':
#         #         model_name = 'resnet50_224_25_grad_cam/1'
#         #     else:
#         #         model_name = 'resnet50_224_25_grad_cam/1'

#         #     app.config.update(
#         #         MODEL_CL_PATH=os.path.join(basedir, 'models', 'car_classification', model_name)
#         #     )

#             global MODEL_CL, img_size_cl
#         #     MODEL_CL, img_size_cl = load_model_classification(app.config['MODEL_CL_PATH'])
            
#         else:
#             src_dir = os.path.join(app.config['DEFAULT_UPLOADED_PATH'], app.config['DROPZONE_FILENAME'])
#             dst_dir = os.path.join(app.config['UPLOADED_PATH'], app.config['DROPZONE_FILENAME'])

#             if os.path.isfile(src_dir):
#                 shutil.copy(src_dir, dst_dir)

#     img_path = os.path.join(app.config['UPLOADED_PATH'], app.config['DROPZONE_FILENAME'])

#     default_uploads = os.listdir(os.path.join(app.root_path, 'static', 'img', 'default_uploads'))

#     # current date and time
#     now = datetime.now()
#     timestamp = datetime.timestamp(now)
    
#     folder_name = str(uuid.uuid4().hex)
#     predicted_path = os.path.join(app.config['PREDICTED_PATH'], folder_name)
#     os.makedirs(predicted_path)

#     app.config.update(
#         PREVIOUS_PREDICTED_PATH=predicted_path,
#     )

#     images, keep_class = object_detection_api(MODEL_OD, MODEL_CL, img_size_od, img_size_cl, 
#                             img_path, predicted_path=predicted_path, threshold=0.6, rect_th=2, text_size=2, text_th=2)

#     os.remove(img_path)

#     return render_template('predict.html', images=images, classes=keep_class, default_uploads=default_uploads)

if __name__ == '__main__':
    app.run()
