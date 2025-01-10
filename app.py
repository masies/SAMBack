from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sam-front-sigma.vercel.app/",
        "https://sam-front-git-main-1024masis-projects.vercel.app/",
        "https://sam-front-1024masis-projects.vercel.app/", 
        "http://localhost:5173",  # For local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model and scaler
scaler_path = os.getenv('SCALER_PATH', './scaler.pkl')
model_path = os.getenv('MODEL_PATH', './model.pkl')

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Column names for the features
FEATURE_COLUMNS = ['Anello', 'A2', 'P2', 'Ratio', 'SIVC', 'Angolo', 'IVS', 'EDD']

@app.post("/api/predict")
async def predict(data: dict):
    try:
        # Prepare the input data in the correct format
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

        return {"prediction": prediction, "scaled_features": scaled_features.tolist()[0]}
    except Exception as e:
        return {"error": str(e), "message": "Error processing prediction"}

@app.get("/api/status")
async def status():
    return "alive"
