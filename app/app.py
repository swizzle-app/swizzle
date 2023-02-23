                         ###################
                        #                  #
 #######               #  #  #####  #####  #   ###
#       #      #      #   #     #      #   #  #   #
 ###     #    # #    #    #    #      #    #  ####
    #     #  #   #  #     #   #      #     #  #
####       ##     ##      #  #####  #####  #   ###


#############################################
#                   IMPORTS                 #
#############################################
from flask import Flask, render_template, request
from tensorflow import keras 
import librosa
import numpy as np

import filetype as ft

from preprocessing.prepro import PreProcessor
from postprocessing.postpro import PostProcessor
from plotting.plot import Plotter


app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        data = {}
        return render_template('index.html', data=data)
    elif request.method == 'POST':
        t = request.form['dm_select']
        data = {'dm': t}
        return render_template('index.html', data=data)


@app.route('/about', methods=['GET'])
def about():
    data = {'dm': 'false'}
    return render_template('about.html', data=data)


@app.route('/swizzle', methods=['POST'])
def swizzle():
    if request.files.get("uploader").filename == "":
        return "No file uploaded!"

    f = request.files["uploader"]
    t = request.form["dm_select"]
    
    # Check file-type
    if not ft.guess(f).extension == "wav":
        data = {"dm": t,
                "filename": f.filename}
        return render_template("no_wav.html", data=data)

    # Pre-Processing
    p = PreProcessor()
    audio, _ = librosa.load(f, sr=22050, dtype=np.float32)
    p.preprocess_audio(audio, training=True)
    X = p.output['windows']

    # Loading the model
    swizzle_model = keras.models.load_model("app/model/swizzle_model", compile=False)
    y_pred = swizzle_model.predict(X)

    # Post-Processing
    postpro = PostProcessor()
    post_pro_output = postpro.postprocess_data(y_pred, remove_duplicates=True)

    # Plotting
    plotter = Plotter()
    tabs = plotter.make_plots(post_pro_output)

    data = {'dm': t,
            'filename': f.filename,
            'tabs': tabs}


    # return post_pro_output.tolist()
    return render_template('results.html', data=data)


if __name__ == '__main__':
    app.run()