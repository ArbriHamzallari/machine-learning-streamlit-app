"""Feature engineering utilities.

This module converts raw crime events into a supervised ML dataset.
"""

from __future__ import annotations

import pandas as pd


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-derived columns:
    - hour (0–23)
    - day_of_week (0–6, Monday=0)
    - month (1–12)
    - date_day (date floored to day)
    """
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("Expected column 'date' in df")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["hour"] = out["date"].dt.hour
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["date_day"] = out["date"].dt.floor("D")
    return out


def make_zone_time_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw events to (community_area, date_day).

    Group by:
      community_area, date_day
    Compute:
      - incident_count (number of rows)
      - arrest_rate (mean of arrest)
      - domestic_rate (mean of domestic)
      - top_primary_type (mode of primary_type)
    """
    if "date_day" not in df.columns:
        df = add_time_columns(df)

    def mode_or_na(s: pd.Series):
        m = s.mode(dropna=True)
        if len(m) == 0:
            return pd.NA
        return m.iloc[0]

    grouped = (
        df.groupby(["community_area", "date_day"], as_index=False)
        .agg(
            incident_count=("date_day", "size"),
            arrest_rate=("arrest", "mean"),
            domestic_rate=("domestic", "mean"),
            top_primary_type=("primary_type", mode_or_na),
        )
        .sort_values(["community_area", "date_day"])
        .reset_index(drop=True)
    )
    return grouped


def add_history_features(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    For each community_area:
    - past_1d_count = incident_count shifted by 1 day
    - past_7d_count = rolling sum of previous 7 days (exclude current day)
    - past_14d_count = rolling sum of previous 14 days (exclude current day)
    - trend_7d = past_7d_count - past_14d_count
    - ratio_1d_7d = past_1d_count / (past_7d_count + 1)
    - Fill missing history with 0
    """
    required = {"community_area", "date_day", "incident_count"}
    missing = sorted(required - set(df_agg.columns))
    if missing:
        raise ValueError(f"df_agg missing required columns: {missing}")

    out = df_agg.copy()
    out["date_day"] = pd.to_datetime(out["date_day"], errors="coerce")

    # Build a complete daily index per zone so "past_1d_count" truly means previous day.
    per_zone = []
    for zone, zdf in out.groupby("community_area", sort=False):
        zdf = zdf.sort_values("date_day").set_index("date_day")
        if zdf.index.isna().any():
            zdf = zdf[~zdf.index.isna()]
        if len(zdf.index) == 0:
            continue

        full_idx = pd.date_range(zdf.index.min(), zdf.index.max(), freq="D")
        zdf = zdf.reindex(full_idx)
        zdf["community_area"] = zone

        # Fill metrics for days with no incidents
        zdf["incident_count"] = zdf["incident_count"].fillna(0).astype(int)
        if "arrest_rate" in zdf.columns:
            zdf["arrest_rate"] = zdf["arrest_rate"].fillna(0.0)
        if "domestic_rate" in zdf.columns:
            zdf["domestic_rate"] = zdf["domestic_rate"].fillna(0.0)

        # History features (exclude current day)
        prev = zdf["incident_count"].shift(1).fillna(0)
        zdf["past_1d_count"] = prev.astype(int)
        zdf["past_7d_count"] = prev.rolling(window=7, min_periods=1).sum().fillna(0).astype(int)
        zdf["past_14d_count"] = prev.rolling(window=14, min_periods=1).sum().fillna(0).astype(int)

        # Derived features
        zdf["trend_7d"] = (zdf["past_7d_count"] - zdf["past_14d_count"]).fillna(0)
        zdf["ratio_1d_7d"] = (zdf["past_1d_count"] / (zdf["past_7d_count"] + 1)).fillna(0)

        zdf = zdf.reset_index().rename(columns={"index": "date_day"})
        per_zone.append(zdf)

    if not per_zone:
        out["past_1d_count"] = 0
        out["past_7d_count"] = 0
        out["past_14d_count"] = 0
        out["trend_7d"] = 0
        out["ratio_1d_7d"] = 0
        return out

    result = pd.concat(per_zone, ignore_index=True).sort_values(["community_area", "date_day"]).reset_index(drop=True)
    return result


def add_risk_labels(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Add quantile-based risk labels from incident_count.

    For each community_area independently:
    - Compute 50th (median) and 80th percentiles of incident_count for that zone
    - Assign risk_class relative to that zone's own distribution:
      0 = low risk (incident_count <= 50th percentile)
      1 = medium risk (incident_count <= 80th percentile)
      2 = high risk (incident_count > 80th percentile)
    
    Zones with very few samples (< 3) fall back to global quantiles.
    - Column name: risk_class
    """
    if "incident_count" not in df_agg.columns:
        raise ValueError("df_agg must contain 'incident_count'")

    out = df_agg.copy()
    
    # Compute global quantiles as fallback for zones with few samples
    global_low_threshold = out["incident_count"].quantile(0.50)
    global_high_threshold = out["incident_count"].quantile(0.80)
    
    # Minimum samples needed to compute meaningful percentiles
    min_samples = 3
    
    # Initialize risk_class column (will be set per zone)
    out["risk_class"] = 0
    
    # Process each community_area independently
    for zone, zdf in out.groupby("community_area", sort=False):
        zone_mask = out["community_area"] == zone
        
        # Check if zone has enough samples
        if len(zdf) < min_samples:
            # Fall back to global quantiles
            low_threshold = global_low_threshold
            high_threshold = global_high_threshold
        else:
            # Compute zone-specific quantiles
            low_threshold = zdf["incident_count"].quantile(0.50)
            high_threshold = zdf["incident_count"].quantile(0.80)
        
        # Assign risk_class based on thresholds for this zone
        # risk_class = 0 if incident_count <= low_threshold
        out.loc[zone_mask & (out["incident_count"] <= low_threshold), "risk_class"] = 0
        # risk_class = 1 if incident_count > low_threshold and <= high_threshold
        out.loc[zone_mask & (out["incident_count"] > low_threshold) & (out["incident_count"] <= high_threshold), "risk_class"] = 1
        # risk_class = 2 if incident_count > high_threshold
        out.loc[zone_mask & (out["incident_count"] > high_threshold), "risk_class"] = 2
    
    out["risk_class"] = out["risk_class"].astype(int)
    return out


def build_ml_dataset(df_raw: pd.DataFrame):
    """
    Build supervised ML dataset.

    Returns:
    - X (DataFrame) with columns:
        community_area, day_of_week, month, past_1d_count, past_7d_count, 
        arrest_rate, domestic_rate, trend_7d, ratio_1d_7d
    - y (Series): risk_class
    - meta (DataFrame): community_area, date_day, incident_count
    - df_agg (DataFrame): full aggregated table (with history + risk labels)
    """
    df_time = add_time_columns(df_raw)
    df_agg = make_zone_time_table(df_time)
    df_agg = add_history_features(df_agg)
    df_agg = add_risk_labels(df_agg)

    # Ensure time columns exist for X
    df_agg["day_of_week"] = pd.to_datetime(df_agg["date_day"]).dt.dayofweek
    df_agg["month"] = pd.to_datetime(df_agg["date_day"]).dt.month

    # Ensure derived features exist and fill NaNs with 0
    if "trend_7d" not in df_agg.columns:
        df_agg["trend_7d"] = 0
    if "ratio_1d_7d" not in df_agg.columns:
        df_agg["ratio_1d_7d"] = 0
    df_agg["trend_7d"] = df_agg["trend_7d"].fillna(0)
    df_agg["ratio_1d_7d"] = df_agg["ratio_1d_7d"].fillna(0)

    X_cols = [
        "community_area",
        "day_of_week",
        "month",
        "past_1d_count",
        "past_7d_count",
        "arrest_rate",
        "domestic_rate",
        "trend_7d",
        "ratio_1d_7d",
    ]
    X = df_agg.loc[:, X_cols].copy()
    y = df_agg["risk_class"].copy()
    meta = df_agg.loc[:, ["community_area", "date_day", "incident_count"]].copy()

    assert not X.isna().any().any(), "X contains NaNs"
    assert set(y.dropna().unique()).issubset({0, 1, 2}), "y contains values outside {0,1,2}"

    return X, y, meta, df_agg
