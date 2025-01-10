# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load both the model and scaler when the server starts
model = joblib.load('./model.pkl')
scaler = joblib.load('./scaler.pkl')

# Column names in the correct order for the scaler
FEATURE_COLUMNS = ['Anello', 'A2', 'P2', 'Ratio', 'SIVC', 'Angolo', 'IVS', 'EDD']

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create DataFrame with the input data in the correct format
        input_data = pd.DataFrame({
            'Anello': [float(data['dim_anello'])],
            'A2': [float(data['lunghezza_a2'])],
            'P2': [float(data['lunghezza_p2'])],
            'Ratio': [float(data['rapporto_lam_lpm'])],
            'SIVC': [float(data['distanza_siv_coapt'])],
            'Angolo': [float(data['angolo_ma'])],
            'IVS': [float(data['setto_basale'])],
            'EDD': [float(data['lv_edd'])]
        })[FEATURE_COLUMNS]  # Ensure correct column order
        
        # Scale the features
        scaled_features = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict_proba(scaled_features)[0][1] * 100
        
        return jsonify({
            'prediction': prediction,
            'scaled_features': scaled_features.tolist()[0]  # Include scaled features for debugging
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error processing prediction'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)