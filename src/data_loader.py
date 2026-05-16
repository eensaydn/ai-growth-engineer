"""Load and validate raw CSV inputs."""
from __future__ import annotations

import pandas as pd

from .config import EVENTS_CSV, EVENT_TYPES, PROFILES_CSV
from .utils import get_logger

log = get_logger(__name__)


def load_profiles(path=PROFILES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"user_id", "plan_type", "country", "device_type", "industry"}
    missing = expected - set(df.columns)
    assert not missing, f"Profiles CSV missing columns: {missing}"
    assert df["user_id"].is_unique, "Duplicate user_ids in profiles"
    assert df["user_id"].notna().all(), "Null user_ids in profiles"
    log.info("Loaded profiles: %d users", len(df))
    return df


def load_events(path=EVENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    expected = {"user_id", "event_type", "timestamp"}
    missing = expected - set(df.columns)
    assert not missing, f"Events CSV missing columns: {missing}"
    assert df["user_id"].notna().all(), "Null user_ids in events"
    assert df["timestamp"].notna().all(), "Null timestamps in events"
    unknown = set(df["event_type"].unique()) - set(EVENT_TYPES)
    if unknown:
        log.warning("Unknown event types found: %s", unknown)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded events: %d rows, %s -> %s",
             len(df), df["timestamp"].min(), df["timestamp"].max())
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = load_profiles()
    events = load_events()
    extra = set(events["user_id"]) - set(profiles["user_id"])
    assert not extra, f"Events reference unknown users: {extra}"
    return profiles, events
