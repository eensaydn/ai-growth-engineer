"""Build the supervised target label and rule-based scores (churn, growth)."""
from __future__ import annotations

import pandas as pd

from .config import (
    HIGH_PAYER_PERCENTILE,
    PLAN_GROWTH_MULTIPLIER,
    WINDOW_DAYS,
)
from .utils import get_logger, normalize

log = get_logger(__name__)


def add_is_high_payer(features: pd.DataFrame, percentile: float = HIGH_PAYER_PERCENTILE) -> pd.DataFrame:
    """Top tertile of n_payment_success becomes the positive class.

    Rationale: 99/100 users have at least one payment_success, making
    'ever_paid' degenerate. We predict the more useful business signal:
    is this user a HIGH-volume payer.
    """
    df = features.copy()
    cutoff = df["n_payment_success"].quantile(percentile)
    df["is_high_payer"] = (df["n_payment_success"] >= cutoff).astype(int)
    log.info("is_high_payer cutoff = %.1f payments. Positive class: %d/%d",
             cutoff, df["is_high_payer"].sum(), len(df))
    return df


def compute_churn_risk(features: pd.DataFrame) -> pd.Series:
    f = features
    return (
        0.40 * normalize(f["last_event_day"], 0, WINDOW_DAYS)
        + 0.25 * normalize(f["recency_checkout_days"], 0, WINDOW_DAYS)
        + 0.20 * (1 - normalize(f["activity_trend"], -1, 1))
        + 0.15 * normalize(f["trial_extension_intensity"], 0, 0.3)
    )


def compute_growth_potential(features: pd.DataFrame) -> pd.Series:
    f = features
    base_engagement = (
        0.30 * normalize(f["n_login"], 0, 15)
        + 0.30 * normalize(f["n_feature_click"], 0, 10)
        + 0.20 * normalize(f["n_checkout_start"], 0, 9)
        + 0.20 * (1 - normalize(f["last_event_day"], 0, WINDOW_DAYS))
    )
    plan_mult = f["plan_type"].map(PLAN_GROWTH_MULTIPLIER).fillna(0.0).to_numpy()
    return pd.Series(base_engagement * plan_mult, index=f.index)


def add_scores(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["churn_risk"] = compute_churn_risk(df).round(4)
    df["growth_potential"] = compute_growth_potential(df).round(4)
    log.info("Scores added. churn_risk mean=%.3f, growth_potential mean=%.3f",
             df["churn_risk"].mean(), df["growth_potential"].mean())
    return df
