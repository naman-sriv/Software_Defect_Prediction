import pickle
from flask import Flask, render_template, request, app, jsonify, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
## Load the model
model = pickle.load(open('stacked_clf.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', method=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = (np.array(list(data.values())).reshape(1,-1))
    output =  model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)