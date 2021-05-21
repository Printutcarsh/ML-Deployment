from flask import Flask, render_template, request, url_for, flash, redirect
import os
from utils import extract_text, detect_object

app = Flask(__name__)

app.config['SECRET_KEY'] = '075427478c7d6b46f6faf44b1dbf9b55'

@app.route('/')
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
                flash(f"{x[0].capitalize()}, Probability Score - {x[1]:.3f}",'primary')
        return redirect(url_for('detect'))
    else:
        return render_template('object_detect.html')


if __name__ == '__main__':
    app.run(debug=False)
