from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize the FastAPI app
app = FastAPI()

# CORS configuration to allow specific origins
origins = [
    "https://sam-front-sigma.vercel.app/", 
    "https://sam-front-git-main-1024masis-projects.vercel.app/", 
    "https://sam-front-1024masis-projects.vercel.app/", 
    "http://localhost:5173",  # For local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input validation
class PredictionRequest(BaseModel):
    dim_anello: float
    lunghezza_a2: float
    lunghezza_p2: float
    rapporto_lam_lpm: float
    distanza_siv_coapt: float
    angolo_ma: float
    setto_basale: float
    lv_edd: float

# Load both the model and scaler when the server starts
scaler_path = os.getenv('SCALER_PATH', './scaler.pkl')
model_path = os.getenv('MODEL_PATH', './model.pkl')

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Column names in the correct order for the scaler
FEATURE_COLUMNS = ['Anello', 'A2', 'P2', 'Ratio', 'SIVC', 'Angolo', 'IVS', 'EDD']

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI on Render!"}

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    try:
        # Create DataFrame with the input data in the correct format
        input_data = pd.DataFrame({
            'Anello': [request.dim_anello],
            'A2': [request.lunghezza_a2],
            'P2': [request.lunghezza_p2],
            'Ratio': [request.rapporto_lam_lpm],
            'SIVC': [request.distanza_siv_coapt],
            'Angolo': [request.angolo_ma],
            'IVS': [request.setto_basale],
            'EDD': [request.lv_edd]
        })[FEATURE_COLUMNS]  # Ensure correct column order
        
        # Scale the features
        scaled_features = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict_proba(scaled_features)[0][1] * 100
        
        return {
            'prediction': prediction,
            'scaled_features': scaled_features.tolist()[0]  # Include scaled features for debugging
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")

@app.get("/api/status")
async def status():
    return {"status": "alive"}

