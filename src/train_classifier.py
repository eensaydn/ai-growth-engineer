"""Train RandomForest classifier for high-payer prediction with 5-fold CV."""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import (
    CONFUSION_MATRIX_PNG,
    CV_FOLDS,
    EVALUATION_REPORT,
    FEATURE_IMPORTANCES_CSV,
    LEAKAGE_COLS,
    MODELS_DIR,
    RANDOM_STATE,
    RF_MODEL_PATH,
    RF_PARAMS,
)
from .utils import ensure_dir, get_logger

log = get_logger(__name__)


# Columns to drop from the feature matrix (in addition to LEAKAGE_COLS).
# These are id/string columns kept in the features table for traceability
# but never fed to the model.
NON_FEATURE_COLS = ["user_id", "plan_type", "country", "device_type", "industry"]


def build_feature_matrix(features: pd.DataFrame):
    drop_cols = [c for c in NON_FEATURE_COLS + LEAKAGE_COLS if c in features.columns]
    X = features.drop(columns=drop_cols)
    X = X.select_dtypes(include="number")
    leak = [c for c in X.columns if c in LEAKAGE_COLS]
    assert not leak, f"Leakage columns still in feature matrix: {leak}"
    return X


def train(features_with_target: pd.DataFrame) -> dict:
    X = build_feature_matrix(features_with_target)
    y = features_with_target["is_high_payer"].astype(int).to_numpy()
    feature_names = list(X.columns)
    log.info("Feature matrix: %s, target positives: %d/%d", X.shape, int(y.sum()), len(y))

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rf_for_cv = RandomForestClassifier(**RF_PARAMS)
    y_pred = cross_val_predict(rf_for_cv, X, y, cv=skf, n_jobs=-1)
    y_proba = cross_val_predict(rf_for_cv, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]

    metrics = {
        "accuracy": float(np.mean(y_pred == y)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred)),
    }
    log.info("CV metrics: %s", {k: round(v, 3) for k, v in metrics.items()})

    # Final model on all data
    final = RandomForestClassifier(**RF_PARAMS)
    final.fit(X, y)

    # Persist artefacts
    ensure_dir(MODELS_DIR)
    joblib.dump({"model": final, "feature_names": feature_names}, RF_MODEL_PATH)
    log.info("Saved RF to %s", RF_MODEL_PATH)

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": final.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importances["rank"] = importances.index + 1
    ensure_dir(FEATURE_IMPORTANCES_CSV.parent)
    importances.to_csv(FEATURE_IMPORTANCES_CSV, index=False)

    _save_confusion_matrix(y, y_pred)
    _save_eval_report(y, y_pred, metrics, importances)

    return {
        "metrics": metrics,
        "feature_names": feature_names,
        "importances": importances,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


def _save_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["regular", "high_payer"]).plot(ax=ax, colorbar=False)
    ax.set_title("RF Confusion Matrix (5-fold CV)")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=120)
    plt.close(fig)


def _save_eval_report(y_true, y_pred, metrics: dict, importances: pd.DataFrame) -> None:
    lines = ["=== Random Forest Classifier — 5-fold CV ===\n"]
    for k, v in metrics.items():
        lines.append(f"{k:>10}: {v:.4f}")
    lines.append("\nClassification report:\n")
    lines.append(classification_report(y_true, y_pred, target_names=["regular", "high_payer"]))
    lines.append("\nTop 10 feature importances:\n")
    lines.append(importances.head(10).to_string(index=False))
    EVALUATION_REPORT.write_text("\n".join(lines))


def save_classifier_metadata(metrics: dict, feature_names: list, path) -> None:
    path = ensure_dir(path.parent) / path.name
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update({
        "classifier": {
            "metrics_cv": metrics,
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "hyperparameters": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        }
    })
    path.write_text(json.dumps(existing, indent=2))
