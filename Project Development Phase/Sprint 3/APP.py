import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
from gevent.pywsgi import WSGIServer
from flask import send_from_directory
from joblib import Parallel,delayed
import joblib
import pandas as pd
from scipy.sparse import issparse



UPLOAD_FOLDER = 'D:/sdhi/PROJECT DEVELOPMENT PHASE/SPRINT 3/UPLOADS'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        data = [[request.form.get('age'),request.form.get('gender'),request.form.get('tb'),request.form.get('ap'),request.form.get('aa'),request.form.get('asa')
                ,request.form.get('tp'),request.form.get('a'),request.form.get('agr')]]
  
        df = pd.DataFrame(data, columns=['Age','Gender','Total_Bilirubin','Alkaline_Phosphotase',
                                         'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens',
                                         'Albumin','Albumin_and_Globulin_Ratio'])

        gh=joblib.load('Liver.pkl')
        num=gh.predict(df)
        if num[0]==1:
            k="The patient has high probability of a Liver Diesease."
        else:
            k="The patient has low probability of a Liver Diesease."
        return render_template('predict.html', num=k)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)