from __future__ import annotations

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


app = Flask(__name__)

# ====== CONFIG ======
MODEL_PATH = Path("model") / "prodtaken.pkl"

FEATURES = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

# Default values (biar GET/POST tidak perlu kirim semua)
DEFAULTS = {
    "Age": 30,
    "TypeofContact": "Company Invited",
    "CityTier": 1,
    "DurationOfPitch": 10.0,
    "Occupation": "Salaried",
    "Gender": "Male",
    "NumberOfPersonVisiting": 2.0,
    "NumberOfFollowups": 1.0,
    "ProductPitched": "Basic",
    "PreferredPropertyStar": 3.0,
    "MaritalStatus": "Unmarried",
    "NumberOfTrips": 1.0,
    "Passport": 0,
    "PitchSatisfactionScore": 3,
    "OwnCar": 0,
    "NumberOfChildrenVisiting": 0.0,
    "Designation": "Manager",
    "MonthlyIncome": 20000.0,
}

NUM_INT = {"Age", "CityTier", "Passport", "PitchSatisfactionScore", "OwnCar"}
NUM_FLOAT = {
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
}

CAT = set(FEATURES) - NUM_INT - NUM_FLOAT


# ====== LOAD MODEL ======
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()


# ====== HELPERS ======
def _to_int(v, default: int) -> int:
    if v is None or v == "":
        return int(default)
    return int(v)

def _to_float(v, default: float) -> float:
    if v is None or v == "":
        return float(default)
    return float(v)

def build_input_from_get(args) -> dict:
    data = {}
    for k in FEATURES:
        if k in NUM_INT:
            data[k] = _to_int(args.get(k), DEFAULTS[k])
        elif k in NUM_FLOAT:
            data[k] = _to_float(args.get(k), DEFAULTS[k])
        else:
            # categorical
            val = args.get(k)
            data[k] = str(val) if (val is not None and val != "") else str(DEFAULTS[k])
    return data

def build_input_from_json(payload: dict) -> dict:
    data = {}
    for k in FEATURES:
        val = payload.get(k, None)
        if k in NUM_INT:
            data[k] = _to_int(val, DEFAULTS[k])
        elif k in NUM_FLOAT:
            data[k] = _to_float(val, DEFAULTS[k])
        else:
            data[k] = str(val) if (val is not None and val != "") else str(DEFAULTS[k])
    return data

def to_dataframe(data: dict) -> pd.DataFrame:
    # pastikan urutan kolom konsisten
    df = pd.DataFrame([[data[c] for c in FEATURES]], columns=FEATURES)
    return df

def predict_one(df: pd.DataFrame) -> int:
    pred = model.predict(df)
    pred_value = int(np.atleast_1d(pred)[0])  # aman scalar/array
    return pred_value


# ====== ROUTES ======
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "features": len(FEATURES),
    }), 200


@app.route("/predict", methods=["GET"])
def predict_get():
    try:
        data = build_input_from_get(request.args)
        df = to_dataframe(data)
        y = predict_one(df)

        return jsonify({
            "status": "success",
            "input": data,
            "prediction": y
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400


@app.route("/predict_json", methods=["POST"])
def predict_post():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"status": "error", "message": "JSON body must be an object/dict"}), 400

        data = build_input_from_json(payload)
        df = to_dataframe(data)
        y = predict_one(df)

        return jsonify({
            "status": "success",
            "input": data,
            "prediction": y
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
