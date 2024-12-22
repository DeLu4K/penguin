import pickle
from flask import Flask, jsonify, request
import numpy as np
import os

species = ['Chinstrap', 'Adélie', 'Gentoo']

def validate_input(pengu):
    required_keys = ['island', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    for key in required_keys:
        if key not in pengu:
            return False, f"Missing key: {key}"
    return True, None

def predict_single(penguin, dv, scaler, model):
    categorics = dv.transform([{
        'island': penguin['island'],
        'sex': penguin['sex']
    }])
    
    numerics = scaler.transform([[
        penguin['bill_length_mm'],
        penguin['bill_depth_mm'],
        penguin['flipper_length_mm'],
        penguin['body_mass_g']
    ]])
    
    y_pred = model.predict(numerics)[0]
    combined_feat = np.hstack([categorics, numerics])
    y_prob = model.predict_proba(combined_feat)[0][y_pred]
    return(y_pred, y_prob)

def predict(scaler, model):
    pengu = request.get_json()
    is_valid, error = validate_input(pengu)
    if not is_valid:
        return jsonify({'error': error}), 400
    especie, prob = predict_single(pengu, scaler, model)
    
    result = {
        'pinguino': species[especie],
        'probabilidad': float(prob)
    }
    return jsonify(result)

app = Flask('penguin')


@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('../models/lr.pck', 'rb') as f:
        scaler, model_lr = pickle.load(f)
    return predict(scaler, model_lr)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('../models/svm.pck', 'rb') as f:
        scaler, model_svm = pickle.load(f)
    return predict(scaler, model_svm)

@app.route('/predict_dt', methods=['POST'])
def predict_tree():
    with open('../models/dt.pck', 'rb') as f:
        scaler, model_tree = pickle.load(f)
    return predict(scaler,model_tree)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('../models/knn.pck', 'rb') as f:
        scaler, model_knn = pickle.load(f)
    return predict(scaler,model_knn)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
