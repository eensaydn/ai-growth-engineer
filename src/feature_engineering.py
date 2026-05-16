"""Per-user feature engineering from raw event log."""
from __future__ import annotations

import pandas as pd

from .config import (
    EVENT_TYPES,
    PLAN_ORDINAL,
    RECENT_WINDOW_DAYS,
    USER_FEATURES_CSV,
    WINDOW_DAYS,
)
from .utils import ensure_dir, get_logger

log = get_logger(__name__)


def _event_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Wide table of event counts per user (one column per event type)."""
    counts = (
        events.groupby(["user_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=EVENT_TYPES, fill_value=0)
    )
    counts.columns = [f"n_{c}" for c in counts.columns]
    counts["n_total_events"] = counts.sum(axis=1)
    return counts.reset_index()


def _temporal_features(events: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Recency, tenure, days active per user."""
    grp = events.groupby("user_id")
    first_ts = grp["timestamp"].min()
    last_ts = grp["timestamp"].max()
    days_active = events.assign(d=events["timestamp"].dt.normalize()).groupby("user_id")["d"].nunique()

    out = pd.DataFrame({
        "first_event_day": (reference_date - first_ts).dt.total_seconds() / 86400.0,
        "last_event_day": (reference_date - last_ts).dt.total_seconds() / 86400.0,
        "days_active": days_active,
    })
    out["tenure_days"] = (last_ts - first_ts).dt.total_seconds() / 86400.0 + 1.0

    payment_last = events[events["event_type"] == "payment_success"].groupby("user_id")["timestamp"].max()
    checkout_last = events[events["event_type"] == "checkout_start"].groupby("user_id")["timestamp"].max()
    out["recency_payment_days"] = ((reference_date - payment_last).dt.total_seconds() / 86400.0).reindex(out.index).fillna(WINDOW_DAYS)
    out["recency_checkout_days"] = ((reference_date - checkout_last).dt.total_seconds() / 86400.0).reindex(out.index).fillna(WINDOW_DAYS)

    return out.reset_index()


def _activity_trend(events: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """events_last_7d, events_first_7d, activity_trend per user."""
    cutoff_last = reference_date - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    last7 = events[events["timestamp"] >= cutoff_last].groupby("user_id").size().rename("events_last_7d")

    first_ts = events.groupby("user_id")["timestamp"].min()
    e = events.merge(first_ts.rename("first_ts"), left_on="user_id", right_index=True)
    e_first = e[e["timestamp"] < e["first_ts"] + pd.Timedelta(days=RECENT_WINDOW_DAYS)]
    first7 = e_first.groupby("user_id").size().rename("events_first_7d")

    user_ids = events["user_id"].unique()
    df = pd.DataFrame({"user_id": user_ids}).set_index("user_id")
    df["events_last_7d"] = last7.reindex(df.index).fillna(0).astype(int)
    df["events_first_7d"] = first7.reindex(df.index).fillna(0).astype(int)
    denom = df["events_first_7d"].clip(lower=1)
    df["activity_trend"] = (df["events_last_7d"] - df["events_first_7d"]) / denom
    df["activity_trend"] = df["activity_trend"].clip(-5, 5)
    return df.reset_index()


def _ratios_and_patterns(counts: pd.DataFrame) -> pd.DataFrame:
    """Conversion ratios and behavioral pattern features."""
    df = counts.copy()
    df["checkout_to_payment_rate"] = df["n_payment_success"] / df["n_checkout_start"].clip(lower=1)
    df["login_to_payment_rate"] = df["n_payment_success"] / df["n_login"].clip(lower=1)
    df["payment_per_event"] = df["n_payment_success"] / df["n_total_events"].clip(lower=1)
    df["feature_click_rate"] = df["n_feature_click"] / df["n_total_events"].clip(lower=1)
    df["login_logout_ratio"] = df["n_login"] / df["n_logout"].clip(lower=1)
    df["trial_extension_intensity"] = df["n_trial_extension"] / df["n_total_events"].clip(lower=1)
    return df[[
        "user_id",
        "checkout_to_payment_rate", "login_to_payment_rate",
        "payment_per_event", "feature_click_rate",
        "login_logout_ratio", "trial_extension_intensity",
    ]]


def _encode_profile(profiles: pd.DataFrame) -> pd.DataFrame:
    df = profiles.copy()
    df["plan_ordinal"] = df["plan_type"].map(PLAN_ORDINAL).astype(int)
    df["device_is_desktop"] = (df["device_type"] == "Desktop").astype(int)
    country_oh = pd.get_dummies(df["country"], prefix="country").astype(int)
    industry_oh = pd.get_dummies(df["industry"], prefix="industry").astype(int)
    return pd.concat([
        df[["user_id", "plan_type", "country", "device_type", "industry",
            "plan_ordinal", "device_is_desktop"]],
        country_oh, industry_oh,
    ], axis=1)


def build_user_features(profiles: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join all per-user feature groups into a single table indexed by user_id."""
    reference_date = events["timestamp"].max().normalize() + pd.Timedelta(days=1)
    log.info("REFERENCE_DATE = %s", reference_date.date())

    counts = _event_counts(events)
    temporal = _temporal_features(events, reference_date)
    trend = _activity_trend(events, reference_date)
    ratios = _ratios_and_patterns(counts)
    profile_enc = _encode_profile(profiles)

    feat = (
        profile_enc
        .merge(counts, on="user_id", how="left")
        .merge(temporal, on="user_id", how="left")
        .merge(trend, on="user_id", how="left")
        .merge(ratios, on="user_id", how="left")
    )

    # Users that exist in profiles but have no events: fill numeric NaNs with 0.
    num_cols = feat.select_dtypes(include="number").columns
    feat[num_cols] = feat[num_cols].fillna(0)

    feat["avg_events_per_active_day"] = feat["n_total_events"] / feat["days_active"].clip(lower=1)

    log.info("Built features: %d users x %d columns", len(feat), feat.shape[1])
    return feat


def save_user_features(features: pd.DataFrame, path=USER_FEATURES_CSV) -> None:
    ensure_dir(path.parent)
    features.to_csv(path, index=False)
    log.info("Saved features to %s", path)


if __name__ == "__main__":
    from .data_loader import load_all
    p, e = load_all()
    feats = build_user_features(p, e)
    save_user_features(feats)
    print(feats.head())
    print("Shape:", feats.shape)
