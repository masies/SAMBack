from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
from marshmallow import Schema, fields, ValidationError
from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": [
    "https://sam-front-sigma.vercel.app/", 
    "https://sam-front-git-main-1024masis-projects.vercel.app/", 
    "https://sam-front-1024masis-projects.vercel.app/", 
    "http://localhost:5173",  # For local dev
]}})


Talisman(app)  # Add security headers

class PredictionSchema(Schema):
    dim_anello = fields.Float(required=True)
    lunghezza_a2 = fields.Float(required=True)
    lunghezza_p2 = fields.Float(required=True)
    rapporto_lam_lpm = fields.Float(required=True)
    distanza_siv_coapt = fields.Float(required=True)
    angolo_ma = fields.Float(required=True)
    setto_basale = fields.Float(required=True)
    lv_edd = fields.Float(required=True)

schema = PredictionSchema()

# Load both the model and scaler when the server starts
scaler_path = os.getenv('SCALER_PATH', './scaler.pkl')
model_path = os.getenv('MODEL_PATH', './model.pkl')

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Column names in the correct order for the scaler
FEATURE_COLUMNS = ['Anello', 'A2', 'P2', 'Ratio', 'SIVC', 'Angolo', 'IVS', 'EDD']

@app.route('/')
def home():
    return "Hello, Flask on Render!"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = schema.load(request.json)
        
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

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return  "alive"

# Update this to bind Flask to '0.0.0.0' and the 'PORT' environment variable
if __name__ == '__main__':
    host = '0.0.0.0'  # Bind to all interfaces
    port = int(os.environ.get('PORT', 5000))  # Use the dynamic port from Render
    app.run(host=host, port=port, debug=True)
