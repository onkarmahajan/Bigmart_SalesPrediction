import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import os

app = Flask(__name__)
images_folder = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = images_folder
LinearModel = pickle.load(open('LinearModel.pkl', 'rb'))
DecisionTree = pickle.load(open('DecisionTree.pkl', 'rb'))
RandomForest = pickle.load(open('RandomForest.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/ajaxtrial', methods=['POST'])
def ajaxtrial():
    name = request.form['name']
    newName = name[::-1]

    return jsonify({
        'name' : newName
    })

@app.route('/predict', methods=['POST'])
def predict():
    dict_encoded = {"Low Fat" : [1, 0, 0], "Regular" : [0, 0, 1], "Non Edibile" : [0, 1, 0], "DR" : [1, 0, 0], "FD" : [0, 1, 0], "NC" : [0, 0, 1], "OUT010" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "OUT013" : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], "OUT017" : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], "OUT018" : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], "OUT019" : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "OUT027" : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], "OUT035" : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], "OUT045" : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], "OUT046" : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], "OUT049" : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], "High" : [1, 0, 0], "Medium" : [0, 1, 0], "Small" : [0, 0, 1], "Tier 1" : [1, 0, 0], "Tier 2" : [0, 1, 0], "Tier 3" : [0, 0, 1], "Grocery Store" : [1, 0, 0, 0], "Supermarket Type 1" : [0, 1, 0, 0], "Supermarket Type 2" : [0, 0, 1, 0], "Supermarket Type 3" : [0, 0, 0, 1]}
    features = [x for x in request.form.values()]
    features_list = [float(x) for x in features[:4]]

    for x in features[4:-1]:
        features_list.extend(dict_encoded[x])

    LinearPrediction = float(LinearModel.predict([features_list]))
    DecisionTreePrediction = float(DecisionTree.predict([features_list]))
    RandomForestPrediction = float(RandomForest.predict([features_list]))

    model = features[-1]

    if (model == "linear"):
        prediction = LinearPrediction
    elif (model == "decisiontree"):
        prediction = DecisionTreePrediction
    elif (model == "randomforest"):
        prediction = RandomForestPrediction

    full_filename_r2 = os.path.join(app.config['UPLOAD_FOLDER'], 'r2_compare.png')
    full_filename_cv = os.path.join(app.config['UPLOAD_FOLDER'], 'cv_compare.png')
    # return render_template('index.html', prediction_text = "Outlet Sales: ${0:.2f}".format(prediction), prediction_r2_compare = full_filename_r2, prediction_cv_compare = full_filename_cv)

    return jsonify({
        'prediction' : prediction
    })

@app.route('/bigmart')
def bigmart():
    return redirect('https://www.bigmart.com/')

@app.route('/dataexploration')
def dataexploration():
    return render_template('dataexploration.html')

@app.route('/github')
def github():
    return redirect('https://github.com/onkarmahajan/Bigmart_SalesPrediction/')

if __name__ == "__main__":
    app.run(debug = True)
