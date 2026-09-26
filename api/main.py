"""
Concrete Compressive Strength API
----------------------------------
Serves predictions from the real trained model (concrete_model.pkl +
concrete_scaler.pkl), produced by the Concrete_Compressive_Strength notebook.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"cement":300,"blast_furnace_slag":0,"fly_ash":0,"water":190,"superplasticizer":0,"coarse_aggregate":1000,"fine_aggregate":800,"age":28}'
"""

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Concrete Compressive Strength API", version="1.0.0")

# CORS: open for demo purposes. For production, replace "*" with the exact
# origin(s) that will call this API (e.g. your hosted calculator page).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "concrete_model.pkl"
SCALER_PATH = "concrete_scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    model = None
    scaler = None

# Must match the training notebook exactly: same columns, same order,
# same log1p transform, fit on the same scaler.
FEATURE_COLUMNS = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age (day)",
]
LOG_COLUMNS = ["Cement", "Blast Furnace Slag", "Fly Ash", "Superplasticizer", "Age (day)"]


class MixInput(BaseModel):
    cement: float = Field(..., ge=0, description="kg per m3")
    blast_furnace_slag: float = Field(0, ge=0)
    fly_ash: float = Field(0, ge=0)
    water: float = Field(..., gt=0)
    superplasticizer: float = Field(0, ge=0)
    coarse_aggregate: float = Field(..., ge=0)
    fine_aggregate: float = Field(..., ge=0)
    age: float = Field(..., gt=0, description="days")


class PredictionOutput(BaseModel):
    predicted_strength_mpa: float
    grade_label: str


def grade_label(mpa: float) -> str:
    if mpa >= 50:
        return "Very high"
    if mpa >= 35:
        return "High"
    if mpa >= 20:
        return "Standard structural"
    return "Low"


@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "note": "POST /predict with mix quantities to get a strength prediction.",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(mix: MixInput):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model files not found. Run the notebook's save-model cell to "
                "produce concrete_model.pkl and concrete_scaler.pkl, then place "
                "them next to main.py."
            ),
        )

    row = pd.DataFrame(
        [[
            mix.cement,
            mix.blast_furnace_slag,
            mix.fly_ash,
            mix.water,
            mix.superplasticizer,
            mix.coarse_aggregate,
            mix.fine_aggregate,
            mix.age,
        ]],
        columns=FEATURE_COLUMNS,
    )

    row[LOG_COLUMNS] = np.log1p(row[LOG_COLUMNS])
    row_scaled = pd.DataFrame(scaler.transform(row), columns=FEATURE_COLUMNS)

    prediction = float(model.predict(row_scaled)[0])
    prediction = max(0.0, prediction)

    return PredictionOutput(
        predicted_strength_mpa=round(prediction, 2),
        grade_label=grade_label(prediction),
    )
