from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import math
import os
import pandas as pd


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
SAM_PIPELINE_PATH = os.getenv(
    "SAM_PIPELINE_PATH",
    os.path.join(BASE_DIR, "sam_predictor", "sam_pipeline.joblib"),
)
RING_PIPELINE_PATH = os.getenv(
    "RING_PIPELINE_PATH",
    os.path.join(BASE_DIR, "ring_predictor", "ring_pipeline.joblib"),
)

SAM_BAND_LOW = 0.11
SAM_BAND_HIGH = 0.32

NUMERIC_FEATURES = {
    "Pre_EF": (35.0, 88.0),
    "A2_mm": (3.5, 49.0),
    "P2_mm": (0.0, 35.0),
    "ratio_lam_lpm": (0.0, 3.75),
    "SIV_Coapt_mm": (6.0, 51.5),
    "angolo_ma": (65.0, 170.0),
    "setto_basale": (0.0, 24.4),
    "lv_edd": (18.0, 88.0),
    "Altezza_cm": (126.0, 217.0),
    "Peso_Kg": (12.0, 164.0),
    "BSA": (0.85, 2.85),
    "BMI": (10.0, 61.0),
    "Eta": (0.0, 120.0),
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

NUMERIC_ALIASES = {
    "Pre_EF": ("Pre_EF",),
    "A2_mm": ("A2_mm", "lunghezza_a2", "A2"),
    "P2_mm": ("P2_mm", "lunghezza_p2", "P2"),
    "ratio_lam_lpm": ("ratio_lam_lpm", "rapporto_lam_lpm"),
    "SIV_Coapt_mm": ("SIV_Coapt_mm", "SIV-Coapt_mm", "distanza_siv_coapt", "SIVC"),
    "angolo_ma": ("angolo_ma", "Angolo"),
    "setto_basale": ("setto_basale", "IVS"),
    "lv_edd": ("lv_edd", "EDD"),
    "Altezza_cm": ("Altezza_cm",),
    "Peso_Kg": ("Peso_Kg",),
    "BSA": ("BSA",),
    "BMI": ("BMI",),
    "Eta": ("Eta",),
    "Pre_LVESV": ("Pre_LVESV",),
    "Mitrale_AP_mm": ("Mitrale_AP_mm",),
    "mitrale_IC": ("mitrale_IC",),
}

BOOLEAN_FEATURES = (
    "Any_cleft",
    "Any_leaflet_calcification",
    "Any_annular_calcification",
)

REQUIRED_NUMERIC_FEATURES = tuple(NUMERIC_FEATURES.keys())
REQUIRED_CATEGORICAL_FEATURES = tuple(CATEGORICAL_FEATURES.keys())
REQUIRED_BOOLEAN_FEATURES = BOOLEAN_FEATURES


sam_model = joblib.load(SAM_PIPELINE_PATH)
ring_model = joblib.load(RING_PIPELINE_PATH)
SAM_FEATURE_COLUMNS = list(sam_model.feature_names_in_)
RING_FEATURE_COLUMNS = list(ring_model.feature_names_in_)


def _first_present(data: dict, *keys: str):
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _read_optional_float(data: dict, *keys: str):
    value = _first_present(data, *keys)
    if value is None:
        return None

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{keys[0]} must be a finite number")
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

    for feature, aliases in NUMERIC_ALIASES.items():
        value = _read_optional_float(data, *aliases)
        if value is not None:
            _validate_numeric(feature, value)
            payload[feature] = value

    if "ratio_lam_lpm" not in payload:
        a2 = payload.get("A2_mm")
        p2 = payload.get("P2_mm")
        if a2 is not None and p2:
            ratio = a2 / p2
            _validate_numeric("ratio_lam_lpm", ratio)
            payload["ratio_lam_lpm"] = ratio

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


def _bool01(value):
    if value in (True, 1, "1", "true", "True", "si", "sì", "yes"):
        return 1
    if value in (False, 0, "0", "false", "False", "no"):
        return 0
    return math.nan


def _encode_etiology(value):
    normalized = str(value).strip().lower()
    if normalized in ("mix", "1", "myxomatous", "myxomatous disease"):
        return 1
    if normalized in ("fed", "0", "fibroelastic", "fibroelastic deficiency"):
        return 0
    return math.nan


def _encode_sex(value):
    normalized = str(value).strip().upper()
    if normalized in ("M", "MALE", "0"):
        return 0
    if normalized in ("F", "FEMALE", "1"):
        return 1
    return math.nan


def _build_feature_row(payload: dict) -> dict:
    row = {
        "Pre_EF": payload["Pre_EF"],
        "Lunghezza A2_mm": payload["A2_mm"],
        "Lunghezza P2_mm": payload["P2_mm"],
        "Distanza SIV-Coapt_mm": payload["SIV_Coapt_mm"],
        "Angolo M-A_gradi": payload["angolo_ma"],
        "Setto basale_mm": payload["setto_basale"],
        "LV EDD": payload["lv_edd"],
        "Rapporto LAM/LPM": payload["ratio_lam_lpm"],
        "Eziologia_MIX_FED": _encode_etiology(payload.get("Eziologia_MIX_FED")),
        "Prolapse": 0,
        "Flail": 0,
        "Posterior leaflet": 0,
        "Anterior leaflet": 0,
        "Bileaflet": 0,
        "Leaflet_type": -1,
        "Any cleft": _bool01(payload.get("Any_cleft")),
        "Any calcification leaflet": _bool01(
            payload.get("Any_leaflet_calcification")
        ),
        "Any calcification anello": _bool01(
            payload.get("Any_annular_calcification")
        ),
        "Altezza_cm": payload["Altezza_cm"],
        "Peso_Kg": payload["Peso_Kg"],
        "BSA": payload["BSA"],
        "BMI": payload["BMI"],
        "Età": payload["Eta"],
        "Sesso": _encode_sex(payload.get("Sesso")),
        "Pre_LVESV": payload["Pre_LVESV"],
        "Mitrale_AP_mm": payload["Mitrale_AP_mm"],
        "mitrale_IC": payload["mitrale_IC"],
    }

    lesion = payload.get("Prolapse")
    normalized = lesion.strip().lower()
    if normalized == "prolapse":
        row["Prolapse"] = 1
    elif normalized == "flail":
        row["Flail"] = 1

    leaflet = payload.get("Leaflet_involved")
    normalized = str(leaflet).strip().lower()
    if normalized.startswith("post"):
        row["Posterior leaflet"] = 1
        row["Leaflet_type"] = 0
    elif normalized.startswith("ant"):
        row["Anterior leaflet"] = 1
        row["Leaflet_type"] = 1
    elif normalized.startswith("bi"):
        row["Bileaflet"] = 1
        row["Leaflet_type"] = 2

    scallops = payload.get("scallop_involved") or []
    if isinstance(scallops, str):
        scallops = [scallops]
    scallops = {str(scallop).strip().upper() for scallop in scallops}
    for scallop in ("A1", "A2", "A3", "P1", "P2", "P3"):
        row[scallop] = 1 if scallop in scallops else 0

    return row


def _predict_sam(feature_row: dict) -> tuple[float, str]:
    data = pd.DataFrame([feature_row]).reindex(columns=SAM_FEATURE_COLUMNS)
    probability = float(sam_model.predict_proba(data)[0][1])
    risk_band = (
        "alto"
        if probability >= SAM_BAND_HIGH
        else "intermedio"
        if probability >= SAM_BAND_LOW
        else "basso"
    )

    return probability, risk_band


def _predict_ring(feature_row: dict) -> tuple[float, list[int]]:
    data = pd.DataFrame([feature_row]).reindex(columns=RING_FEATURE_COLUMNS)
    predicted_mm = float(ring_model.predict(data)[0])
    rounded_mm = round(predicted_mm)
    plausible_range = [int(rounded_mm - 2), int(rounded_mm + 2)]
    return round(predicted_mm, 1), plausible_range


@app.post("/api/predict")
async def predict(data: dict):
    try:
        payload = _normalize_payload(data)
        feature_row = _build_feature_row(payload)
        sam_probability, risk_band = _predict_sam(feature_row)
        predicted_ring_mm, predicted_ring_plausible_range = _predict_ring(feature_row)

        return {
            "sam_probability": round(sam_probability * 100, 2),
            "risk_band": risk_band,
            "band_cutoffs": [SAM_BAND_LOW, SAM_BAND_HIGH],
            "predicted_ring_mm": predicted_ring_mm,
            "predicted_ring_plausible_range": predicted_ring_plausible_range,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {e}")


@app.get("/api/status")
async def status():
    return {"status": "alive"}
