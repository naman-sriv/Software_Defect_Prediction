import pickle
from flask import Flask, render_template, request, app, jsonify, url_for
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
## Load the model
model = pickle.load(open('stacked_clf.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = (np.array(list(data.values())).reshape(1,-1))
    print(new_data.shape)
    output =  model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)