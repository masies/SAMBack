from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cloudpickle
import math
import os


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sam-front-sigma.vercel.app",
        "https://sam-front-git-main-1024masis-projects.vercel.app",
        "https://sam-front-1024masis-projects.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_PREDICTOR_PATH = os.getenv(
    "SAM_PREDICTOR_PATH",
    os.path.join(BASE_DIR, "sam_predictor", "sam_predictor.pkl"),
)
RING_PREDICTOR_PATH = os.getenv(
    "RING_PREDICTOR_PATH",
    os.path.join(BASE_DIR, "ring_predictor", "ring_predictor.pkl"),
)

NUMERIC_FEATURES = {
    "Pre_EF": (35.0, 88.0),
    "Lunghezza A2_mm": (3.5, 49.0),
    "Lunghezza P2_mm": (0.0, 35.0),
    "Rapporto LAM/LPM": (0.0, 3.75),
    "Distanza SIV-Coapt_mm": (6.0, 51.5),
    "Angolo M-A_gradi": (65.0, 170.0),
    "Setto basale_mm": (0.0, 24.4),
    "LV EDD": (18.0, 88.0),
    "Altezza_cm": (126.0, 217.0),
    "Peso_Kg": (12.0, 164.0),
    "BSA": (0.85, 2.85),
    "BMI": (10.0, 61.0),
    "Età": (0.0, 120.0),
    "Pre_LVESV": (0.0, 160.0),
    "Mitrale_AP_mm": (12.0, 71.0),
    "mitrale_IC": (17.0, 83.0),
}

CATEGORICAL_FEATURES = {
    "Eziologia_MIX_FED": {"Myxomatous Disease", "Fibroelastic Deficiency"},
    "Prolapse": {"Prolapse", "Flail"},
    "Leaflet_involved": {"Posterior", "Anterior", "Bileaflet"},
    "scallop_involved": {"A1", "A2", "A3", "P1", "P2", "P3"},
    "Sesso": {"M", "F"},
}

BOOLEAN_FEATURES = (
    "Any cleft",
    "Any calcification leaflet",
    "Any calcification anello",
)

REQUIRED_NUMERIC_FEATURES = tuple(NUMERIC_FEATURES.keys())
REQUIRED_CATEGORICAL_FEATURES = tuple(CATEGORICAL_FEATURES.keys())
REQUIRED_BOOLEAN_FEATURES = BOOLEAN_FEATURES


def _load_predictor(path: str):
    with open(path, "rb") as file:
        return cloudpickle.load(file)


sam_predictor = _load_predictor(SAM_PREDICTOR_PATH)
ring_predictor = _load_predictor(RING_PREDICTOR_PATH)


def _read_optional_float(data: dict, key: str):
    value = data.get(key)
    if value in (None, ""):
        return None

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{key} must be a finite number")
    return numeric_value


def _validate_numeric(name: str, value: float):
    min_value, max_value = NUMERIC_FEATURES[name]
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be between {min_value:g} and {max_value:g}")


def _normalize_boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "y", "1", "si", "sì"):
            return True
        if normalized in ("false", "no", "n", "0"):
            return False
    raise ValueError(f"Invalid boolean value: {value}")


def _normalize_payload(data: dict) -> dict:
    payload = {}

    for feature in REQUIRED_NUMERIC_FEATURES:
        value = _read_optional_float(data, feature)
        if value is not None:
            _validate_numeric(feature, value)
            payload[feature] = value

    if "Rapporto LAM/LPM" not in payload:
        a2 = payload.get("Lunghezza A2_mm")
        p2 = payload.get("Lunghezza P2_mm")
        if a2 is not None and p2:
            ratio = a2 / p2
            _validate_numeric("Rapporto LAM/LPM", ratio)
            payload["Rapporto LAM/LPM"] = ratio

    for feature in REQUIRED_NUMERIC_FEATURES:
        if feature not in payload:
            raise ValueError(f"Missing required field: {feature}")

    for feature, allowed_values in CATEGORICAL_FEATURES.items():
        value = data.get(feature)
        if value in (None, ""):
            continue

        if feature == "scallop_involved":
            values = value if isinstance(value, list) else [value]
            if not values:
                raise ValueError("Missing required field: scallop_involved")
            invalid_values = [item for item in values if item not in allowed_values]
            if invalid_values:
                raise ValueError(
                    f"{feature} contains invalid value(s): {', '.join(invalid_values)}"
                )
            payload[feature] = values
            continue

        if value not in allowed_values:
            raise ValueError(f"{feature} must be one of: {', '.join(allowed_values)}")
        payload[feature] = value

    for feature in REQUIRED_CATEGORICAL_FEATURES:
        if feature not in payload:
            raise ValueError(f"Missing required field: {feature}")

    for feature in BOOLEAN_FEATURES:
        value = data.get(feature)
        if value not in (None, ""):
            payload[feature] = _normalize_boolean(value)

    for feature in REQUIRED_BOOLEAN_FEATURES:
        if feature not in payload:
            raise ValueError(f"Missing required field: {feature}")

    return payload


def _predict_sam(payload: dict) -> dict:
    result = sam_predictor(payload)
    probability = float(result["probability"])

    return {
        "probability": probability,
        "risk_band": result["risk_band"],
        "band_cutoffs": result.get("band_cutoffs"),
        "threshold": result.get("threshold"),
        "predicted_class": result.get("predicted_class"),
    }


def _predict_ring(payload: dict) -> dict:
    result = ring_predictor(payload)
    predicted_mm = float(result["predicted_mm"])
    plausible_range = result.get("interval_2mm")

    if plausible_range is None:
        rounded_mm = round(predicted_mm)
        plausible_range = [int(rounded_mm - 2), int(rounded_mm + 2)]

    return {
        "predicted_mm": round(predicted_mm, 1),
        "recommended_size": result.get("recommended_size"),
        "plausible_range": plausible_range,
        "prob_within_2mm": result.get("prob_within_2mm"),
    }


@app.post("/api/predict")
async def predict(data: dict):
    try:
        payload = _normalize_payload(data)
        sam_result = _predict_sam(payload)
        ring_result = _predict_ring(payload)

        return {
            "sam_probability": round(sam_result["probability"] * 100, 2),
            "risk_band": sam_result["risk_band"],
            "band_cutoffs": sam_result["band_cutoffs"],
            "sam_threshold": sam_result["threshold"],
            "sam_predicted_class": sam_result["predicted_class"],
            "predicted_ring_mm": ring_result["predicted_mm"],
            "recommended_ring_size": ring_result["recommended_size"],
            "predicted_ring_plausible_range": ring_result["plausible_range"],
            "ring_prob_within_2mm": ring_result["prob_within_2mm"],
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {e}")


@app.get("/api/status")
async def status():
    return {"status": "alive"}
