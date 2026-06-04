from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sam-front-sigma.vercel.app",
        "https://sam-front-git-main-1024masis-projects.vercel.app",
        "https://sam-front-1024masis-projects.vercel.app", 
        "http://localhost:5173",  # For local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model and scaler. Keep default paths relative to this file so the app
# works whether it is started from SAMBack or from the repository root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.getenv("SCALER_PATH", os.path.join(BASE_DIR, "scaler.pkl"))
model_path = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model.pkl"))

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# The current scaler was trained with six columns. If a future scaler carries
# feature_names_in_, trust that metadata and keep the request mapping stable.
DEFAULT_FEATURE_COLUMNS = ["Anello", "Ratio", "SIVC", "Angolo", "IVS", "EDD"]
FEATURE_COLUMNS = list(getattr(scaler, "feature_names_in_", DEFAULT_FEATURE_COLUMNS))


def _read_float(data: dict, *keys: str) -> float:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(keys[0])

@app.post("/api/predict")
async def predict(data: dict):
    try:
        lunghezza_a2 = _read_float(data, "lunghezza_a2", "A2_mm")
        lunghezza_p2 = _read_float(data, "lunghezza_p2", "P2_mm")
        ratio = data.get("rapporto_lam_lpm") or data.get("ratio_lam_lpm")
        if ratio in (None, ""):
            ratio = lunghezza_a2 / lunghezza_p2

        feature_values = {
            "Anello": _read_float(data, "dim_anello", "Anello"),
            "A2": lunghezza_a2,
            "P2": lunghezza_p2,
            "Ratio": float(ratio),
            "SIVC": _read_float(data, "distanza_siv_coapt", "SIV-Coapt_mm"),
            "Angolo": _read_float(data, "angolo_ma", "Angolo"),
            "IVS": _read_float(data, "setto_basale", "IVS"),
            "EDD": _read_float(data, "lv_edd", "EDD"),
        }

        input_data = pd.DataFrame(
            [{column: feature_values[column] for column in FEATURE_COLUMNS}]
        )

        scaled_features = scaler.transform(input_data)

        prediction = model.predict_proba(scaled_features)[0][1] * 100

        return {"prediction": prediction, "scaled_features": scaled_features.tolist()[0]}
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required field: {e.args[0]}",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid numeric value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {e}")

@app.get("/api/status")
async def status():
    return {"status": "alive"}
