"""FastAPI REST service exposing predictions, segments, and model metadata.

Run with:
    python api.py
        or
    uvicorn api:app --host 0.0.0.0 --port 8000

Visit http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import (
    FEATURE_IMPORTANCES_CSV,
    MODEL_METADATA_PATH,
    PREDICTIONS_CSV,
    SEGMENT_SUMMARY_CSV,
)

app = FastAPI(
    title="ikas Growth API",
    description="AI Growth Engineer case study — user predictions & segments. "
                "Designed to back a Retool dashboard.",
    version="1.0.0",
)

# CORS: allow Retool (or any cross-origin client) to call this API.
# In production, restrict allow_origins to your Retool subdomain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail="Predictions not generated yet. Run `python run_pipeline.py` first.",
        )
    return pd.read_csv(PREDICTIONS_CSV)


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "ikas Growth API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/predictions",
            "/predictions/{user_id}",
            "/segments/summary",
            "/feature-importances",
            "/metadata",
            "/dashboard-data",
            "/refresh (POST)",
        ],
    }


@app.get("/health", tags=["meta"])
def health():
    last_updated = None
    if PREDICTIONS_CSV.exists():
        last_updated = datetime.fromtimestamp(PREDICTIONS_CSV.stat().st_mtime).isoformat()
    return {
        "status": "ok" if PREDICTIONS_CSV.exists() else "no-data",
        "predictions_last_updated": last_updated,
    }


@app.get("/predictions", tags=["predictions"])
def list_predictions(
    segment: Optional[str] = Query(None, description="Filter by segment label"),
    country: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    plan_type: Optional[str] = Query(None),
    device_type: Optional[str] = Query(None),
    behavior_cluster_name: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=1000),
):
    """Return predictions with optional filters."""
    df = _require_predictions()
    if segment: df = df[df["segment"] == segment]
    if country: df = df[df["country"] == country]
    if industry: df = df[df["industry"] == industry]
    if plan_type: df = df[df["plan_type"] == plan_type]
    if device_type: df = df[df["device_type"] == device_type]
    if behavior_cluster_name: df = df[df["behavior_cluster_name"] == behavior_cluster_name]
    if limit: df = df.head(limit)
    return {"count": int(len(df)), "users": df.to_dict(orient="records")}


@app.get("/predictions/{user_id}", tags=["predictions"])
def get_user(user_id: str):
    df = _require_predictions()
    rec = df[df["user_id"] == user_id]
    if rec.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return rec.iloc[0].to_dict()


@app.get("/segments/summary", tags=["segments"])
def segment_summary():
    if not SEGMENT_SUMMARY_CSV.exists():
        raise HTTPException(status_code=503, detail="Pipeline not run yet")
    return pd.read_csv(SEGMENT_SUMMARY_CSV).to_dict(orient="records")


@app.get("/segments/counts", tags=["segments"])
def segment_counts():
    df = _require_predictions()
    counts = df["segment"].value_counts().to_dict()
    return {k: int(v) for k, v in counts.items()}


@app.get("/feature-importances", tags=["model"])
def feature_importances(top: int = Query(10, ge=1, le=100)):
    if not FEATURE_IMPORTANCES_CSV.exists():
        raise HTTPException(status_code=503, detail="Pipeline not run yet")
    return pd.read_csv(FEATURE_IMPORTANCES_CSV).head(top).to_dict(orient="records")


@app.get("/metadata", tags=["model"])
def metadata():
    if not MODEL_METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Pipeline not run yet")
    return json.loads(MODEL_METADATA_PATH.read_text())


@app.get("/filters", tags=["meta"])
def filter_options():
    """Distinct values for each filterable column — useful to populate Retool dropdowns."""
    df = _require_predictions()
    cols = ["segment", "country", "industry", "plan_type",
            "device_type", "behavior_cluster_name"]
    return {c: sorted(df[c].dropna().unique().tolist()) for c in cols}


@app.get("/dashboard-data", tags=["dashboard"])
def dashboard_bundle():
    """One-shot bundle for Retool: KPIs + predictions + summary + importances.

    Calling this endpoint replaces 4 separate calls — faster Retool page loads.
    """
    df = _require_predictions()
    summary = pd.read_csv(SEGMENT_SUMMARY_CSV).to_dict(orient="records") \
        if SEGMENT_SUMMARY_CSV.exists() else []
    importances = pd.read_csv(FEATURE_IMPORTANCES_CSV).head(10).to_dict(orient="records") \
        if FEATURE_IMPORTANCES_CSV.exists() else []
    meta = json.loads(MODEL_METADATA_PATH.read_text()) \
        if MODEL_METADATA_PATH.exists() else {}

    return {
        "generated_at": datetime.fromtimestamp(PREDICTIONS_CSV.stat().st_mtime).isoformat(),
        "kpis": {
            "total_users": int(len(df)),
            "high_value": int((df["segment"] == "High Value").sum()),
            "medium_value": int((df["segment"] == "Medium Value").sum()),
            "churn_risk": int((df["segment"] == "Churn Risk").sum()),
            "growth_potential": int((df["segment"] == "Growth Potential").sum()),
        },
        "predictions": df.to_dict(orient="records"),
        "segment_summary": summary,
        "feature_importances": importances,
        "model_metrics_cv": meta.get("classifier", {}).get("metrics_cv", {}),
        "cluster_info": meta.get("clusterer", {}),
    }


class RefreshResponse(BaseModel):
    status: str
    message: str
    duration_seconds: float


@app.post("/refresh", response_model=RefreshResponse, tags=["pipeline"])
def refresh_pipeline():
    """Re-run the full pipeline (retrain models + regenerate outputs).

    Takes ~3 seconds on the current dataset. Long-running — consider async wrappers
    if the dataset grows large.
    """
    import time
    from src.predict import run_full_pipeline
    t0 = time.time()
    try:
        run_full_pipeline(retrain=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    return RefreshResponse(
        status="ok",
        message="Pipeline re-trained and predictions regenerated.",
        duration_seconds=round(time.time() - t0, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
