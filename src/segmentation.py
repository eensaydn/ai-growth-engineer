"""Assign each user to one of 4 final segments using priority-ordered rules."""
from __future__ import annotations

import pandas as pd

from .config import (
    CHURN_THRESHOLD,
    GROWTH_THRESHOLD,
    HIGH_VALUE_PROB_THRESHOLD,
)
from .utils import get_logger

log = get_logger(__name__)

SEGMENTS = ["Churn Risk", "Growth Potential", "High Value", "Medium Value"]


def _row_segment(row: pd.Series, median_payments: float) -> tuple[str, str]:
    """Returns (segment, reason)."""
    if row["churn_risk"] >= CHURN_THRESHOLD:
        return "Churn Risk", f"churn_risk={row['churn_risk']:.2f}>={CHURN_THRESHOLD}"
    if row["growth_potential"] >= GROWTH_THRESHOLD and row["plan_type"] in ("Free", "Pro"):
        return "Growth Potential", (
            f"growth_potential={row['growth_potential']:.2f}>={GROWTH_THRESHOLD} on {row['plan_type']}"
        )
    if row["payment_likelihood"] >= HIGH_VALUE_PROB_THRESHOLD and row["n_payment_success"] >= median_payments:
        return "High Value", (
            f"payment_likelihood={row['payment_likelihood']:.2f}>={HIGH_VALUE_PROB_THRESHOLD}, "
            f"payments={int(row['n_payment_success'])}>=median({median_payments:.0f})"
        )
    return "Medium Value", "fallback (no rule matched higher tier)"


def assign_segments(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """predictions_df must contain: churn_risk, growth_potential, payment_likelihood,
    plan_type, n_payment_success."""
    df = predictions_df.copy()
    median_payments = df["n_payment_success"].median()
    log.info("Median payments per user = %.1f", median_payments)

    seg = df.apply(lambda r: _row_segment(r, median_payments), axis=1)
    df["segment"] = [s[0] for s in seg]
    df["segment_reason"] = [s[1] for s in seg]

    counts = df["segment"].value_counts().reindex(SEGMENTS).fillna(0).astype(int)
    log.info("Segment distribution:\n%s", counts.to_string())
    return df


def segment_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg, sub in predictions_df.groupby("segment"):
        rows.append({
            "segment": seg,
            "n_users": len(sub),
            "avg_payment_likelihood": round(sub["payment_likelihood"].mean(), 3),
            "avg_churn_risk": round(sub["churn_risk"].mean(), 3),
            "avg_growth_potential": round(sub["growth_potential"].mean(), 3),
            "avg_n_payment_success": round(sub["n_payment_success"].mean(), 2),
            "top_plan_type": sub["plan_type"].mode().iat[0] if not sub.empty else None,
            "top_country": sub["country"].mode().iat[0] if not sub.empty else None,
            "top_industry": sub["industry"].mode().iat[0] if not sub.empty else None,
        })
    return pd.DataFrame(rows).set_index("segment").reindex(SEGMENTS).dropna(how="all").reset_index()
