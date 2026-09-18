"""FastAPI REST endpoint (FYP proposal §6, §12.2).

Serves the trained models over HTTP with automatic Swagger documentation.

Run:
    uvicorn app.api:app --reload --port 8000
Then open http://localhost:8000/docs for interactive Swagger UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

# make ``src`` importable when launched via uvicorn from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.models.predict import EXAMPLE_PHONE, get_service  # noqa: E402

app = FastAPI(
    title="Mobile Price Prediction API",
    description=(
        "Predicts a smartphone's price tier (0=Budget … 3=Flagship) and an "
        "estimated price from its hardware specifications, with per-prediction "
        "SHAP explanations.\n\nFYP — University of Education, Lahore (Vehari)."
    ),
    version="1.0.0",
)


class PhoneSpecs(BaseModel):
    """Raw hardware specifications for one phone."""
    battery_power: int = Field(..., ge=500, le=7000, description="Battery capacity (mAh)")
    blue: int = Field(0, ge=0, le=1, description="Bluetooth (0/1)")
    clock_speed: float = Field(..., ge=0.5, le=3.5, description="CPU clock speed (GHz)")
    dual_sim: int = Field(0, ge=0, le=1)
    fc: int = Field(..., ge=0, le=64, description="Front camera (MP)")
    four_g: int = Field(0, ge=0, le=1)
    int_memory: int = Field(..., ge=2, le=1024, description="Internal storage (GB)")
    m_dep: float = Field(..., ge=0.1, le=1.5, description="Mobile depth (cm)")
    mobile_wt: int = Field(..., ge=80, le=350, description="Weight (g)")
    n_cores: int = Field(..., ge=1, le=16, description="CPU cores")
    pc: int = Field(..., ge=0, le=108, description="Primary camera (MP)")
    px_height: int = Field(..., ge=0, le=3000)
    px_width: int = Field(..., ge=300, le=3000)
    ram: int = Field(..., ge=256, le=16384, description="RAM (MB)")
    sc_h: float = Field(..., ge=5, le=25, description="Screen height (cm)")
    sc_w: float = Field(..., ge=0, le=20, description="Screen width (cm)")
    talk_time: int = Field(..., ge=2, le=30)
    three_g: int = Field(0, ge=0, le=1)
    touch_screen: int = Field(1, ge=0, le=1)
    wifi: int = Field(1, ge=0, le=1)

    model_config = {"json_schema_extra": {"example": EXAMPLE_PHONE}}


class Contribution(BaseModel):
    feature: str
    contribution_usd: float


class PredictionResponse(BaseModel):
    price_range: int
    price_tier: str
    class_probabilities: dict[str, float]
    estimated_price_usd: float
    top_contributions: list[Contribution]


@app.get("/", tags=["health"])
def root() -> dict:
    """Health check + basic service metadata."""
    svc = get_service()
    return {
        "status": "ok",
        "service": "mobile-price-prediction",
        "best_classifier": svc.meta["best_classifier"],
        "best_regressor": svc.meta["best_regressor"],
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(specs: PhoneSpecs) -> dict:
    """Predict price tier + estimated price for one phone."""
    try:
        return get_service().predict(specs.model_dump(), explain=True)
    except Exception as exc:  # surface clean errors to the client
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
