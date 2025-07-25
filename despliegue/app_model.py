from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Landing page
@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

# Predict
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    with open('ad_model.pkl', 'rb') as f:
        model = pickle.load(f)

    alcohol = request.args.get('alcohol', None)
    ph = request.args.get('pH', None)
    sulphates = request.args.get('sulphates', None)

    if alcohol is None or ph is None or sulphates is None:
        return "Args empty, not enough data to predict", 400  # Mejor devolver un código HTTP de error

    try:
        features = [float(alcohol), float(ph), float(sulphates)]
    except ValueError:
        return "Invalid input: unable to convert to float", 400

    prediction = model.predict([features])

    return jsonify({'prediction': prediction[0]})

#Retrain


if __name__ == '__main__':
    app.run(debug=True)
