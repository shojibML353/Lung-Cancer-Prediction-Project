from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ==============================
# Load ML model (safe path)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "LGModel.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ heart_model.joblib file not found")

model = joblib.load(MODEL_PATH)

# ==============================
# FastAPI app
# ==============================
app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0"
)

# ==============================
# CORS (Frontend connect)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production এ specific domain দিবে
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Input Schema with STRICT validation
# ==============================

class LungCancerInput(BaseModel):
    gender: int = Field(..., ge=0, le=1, description="Male=1, Female=0")
    age: int = Field(..., gt=0, description="Age must be > 0")
    smoking: int = Field(..., ge=1, le=2)
    yellow_fingers: int = Field(..., ge=1, le=2)
    anxiety: int = Field(..., ge=1, le=2)
    peer_pressure: int = Field(..., ge=1, le=2)
    chronic_disease: int = Field(..., ge=1, le=2)
    fatigue: int = Field(..., ge=1, le=2)
    allergy: int = Field(..., ge=1, le=2)
    wheezing: int = Field(..., ge=1, le=2)
    alcohol_consuming: int = Field(..., ge=1, le=2)
    coughing: int = Field(..., ge=1, le=2)
    shortness_of_breath: int = Field(..., ge=1, le=2)
    swallowing_difficulty: int = Field(..., ge=1, le=2)
    chest_pain: int = Field(..., ge=1, le=2)


    # Blank / empty check
    @field_validator("*", mode="before")
    @classmethod
    def no_blank_value(cls, v):
        if v is None or v == "":
            raise ValueError("Field cannot be blank")
        return v

# ==============================
# Routes
# ==============================
# Get absolute path to frontend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# Serve CSS & JS
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

    
    
@app.post("/predict")
def predict(data: LungCancerInput):

    values = list(data.model_dump().values())


    features = np.array([values])

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "result": "lung cancer Detected" if prediction == 1 else "No lung cancer Detected"
    }
