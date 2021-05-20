from flask import Flask, render_template, request, url_for, flash, redirect
import os
import cv2
import easyocr
import pickle
import numpy as np
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

app = Flask(__name__)

app.config['SECRET_KEY'] = '075427478c7d6b46f6faf44b1dbf9b55'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def extract_text(filepath):
    reader = easyocr.Reader(['en'])
    img = cv2.imread(filepath)
    text = reader.readtext(img, detail=0)
    text = ' '.join(text)
    return text

def detect_object(filepath):
    model = models.load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')
    with open('file.pkb', 'rb') as f:
        labels_to_names = pickle.load(f)
    image = read_image_bgr(filepath)

    image = preprocess_image(image)
    image, scale = resize_image(image)

    _, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    objects = []
    for score, label in zip(scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.4:
            break
        else:
            objects.append((score, labels_to_names[label]))

    return objects


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', file.filename)
        file.save(file_path)
        text = extract_text(file_path)
        if len(text) == 0:
            flash('No text found.','primary')
        else:
            flash(f'The extracted text is :- {text}', 'primary')
        return redirect(url_for('extract'))
    else:
        return render_template('extract.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', file.filename)
        file.save(file_path)
        objects = detect_object(file_path)
        if len(objects) == 0:
            flash('No objects detected.','primary')
        else:
            flash("The predictions are : -",'primary')
            for x in objects:
                flash(f"{x[1].capitalize()}, Probability Score - {x[0]:.3f}",'primary')
        return redirect(url_for('detect'))
    else:
        return render_template('object_detect.html')


if __name__ == '__main__':
    app.run(debug=False)
