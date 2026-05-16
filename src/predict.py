"""End-to-end orchestrator: load -> features -> labels -> models -> segments -> outputs."""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd

from .config import (
    KMEANS_FEATURES,
    KMEANS_MODEL_PATH,
    MODEL_METADATA_PATH,
    PREDICTIONS_CSV,
    PREDICTIONS_JSON,
    RF_MODEL_PATH,
    SCALER_PATH,
    SEGMENT_SUMMARY_CSV,
)
from .data_loader import load_all
from .feature_engineering import build_user_features, save_user_features
from .labeling import add_is_high_payer, add_scores
from .segmentation import SEGMENTS, assign_segments, segment_summary
from .train_classifier import build_feature_matrix, save_classifier_metadata, train as train_rf
from .train_clusterer import train as train_kmeans
from .utils import ensure_dir, get_logger

log = get_logger(__name__)


def run_full_pipeline(retrain: bool = True) -> pd.DataFrame:
    """Run the entire pipeline. Always retrains models for case-study reproducibility."""
    profiles, events = load_all()
    feats = build_user_features(profiles, events)
    feats = add_is_high_payer(feats)
    feats = add_scores(feats)
    save_user_features(feats)

    if retrain:
        log.info("=== Training classifier ===")
        clf_info = train_rf(feats)
        save_classifier_metadata(clf_info["metrics"], clf_info["feature_names"], MODEL_METADATA_PATH)

        log.info("=== Training clusterer ===")
        clu_info = train_kmeans(feats)
    else:
        clf_info = None
        clu_info = None

    log.info("=== Generating predictions ===")
    rf_bundle = joblib.load(RF_MODEL_PATH)
    rf_model = rf_bundle["model"]
    rf_features = rf_bundle["feature_names"]
    X = build_feature_matrix(feats)[rf_features]
    payment_likelihood = rf_model.predict_proba(X)[:, 1]

    km_bundle = joblib.load(KMEANS_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    Xk = scaler.transform(feats[KMEANS_FEATURES].to_numpy())
    cluster_ids = km_bundle["model"].predict(Xk)
    cluster_names = km_bundle["names"]

    out = feats[[
        "user_id", "plan_type", "country", "device_type", "industry",
        "n_payment_success", "n_checkout_start", "n_login", "last_event_day",
        "churn_risk", "growth_potential",
    ]].copy()
    out["payment_likelihood"] = payment_likelihood.round(4)
    out["behavior_cluster"] = cluster_ids
    out["behavior_cluster_name"] = [cluster_names[int(c)] for c in cluster_ids]

    out = assign_segments(out)
    out = out.round({"churn_risk": 4, "growth_potential": 4, "payment_likelihood": 4})

    _write_outputs(out)
    return out


def _write_outputs(predictions: pd.DataFrame) -> None:
    ensure_dir(PREDICTIONS_CSV.parent)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    log.info("Wrote %s", PREDICTIONS_CSV)

    summary = segment_summary(predictions)
    summary.to_csv(SEGMENT_SUMMARY_CSV, index=False)
    log.info("Wrote %s", SEGMENT_SUMMARY_CSV)

    counts = predictions["segment"].value_counts().reindex(SEGMENTS).fillna(0).astype(int).to_dict()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_users": int(len(predictions)),
        "model_version": "v1.0",
        "segment_counts": counts,
        "users": json.loads(predictions.to_json(orient="records")),
    }
    PREDICTIONS_JSON.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", PREDICTIONS_JSON)
