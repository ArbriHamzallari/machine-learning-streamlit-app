"""Planning / allocation utilities.

Purpose:
Convert community areas (zones) into a distance-aware planning space and allocate patrols.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd


def compute_zone_centroids(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean latitude/longitude per community_area.

    Args:
        df_raw: Raw/clean event-level DataFrame containing:
            - community_area
            - latitude
            - longitude

    Returns:
        DataFrame with columns: community_area, latitude, longitude
        (latitude/longitude are centroids: means)
    """
    required = {"community_area", "latitude", "longitude"}
    missing = sorted(required - set(df_raw.columns))
    if missing:
        raise ValueError(f"df_raw missing required columns: {missing}")

    # Drop rows missing geo or zone
    df = df_raw.dropna(subset=["community_area", "latitude", "longitude"]).copy()

    centroids = (
        df.groupby("community_area", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .sort_values("community_area")
        .reset_index(drop=True)
    )
    return centroids


def make_grid_mapping(centroids: pd.DataFrame, grid_size: int = 10) -> Dict[int, Tuple[int, int]]:
    """
    Map each zone centroid to integer grid coordinates.

    - Normalize latitude/longitude into [0, grid_size-1]
    - Assign integer grid coordinates
    - Return dict: zone_id -> (x, y)

    Notes:
    - x corresponds to longitude (east-west)
    - y corresponds to latitude  (north-south)
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    required = {"community_area", "latitude", "longitude"}
    missing = sorted(required - set(centroids.columns))
    if missing:
        raise ValueError(f"centroids missing required columns: {missing}")

    df = centroids.dropna(subset=["community_area", "latitude", "longitude"]).copy()

    # Ensure community_area is int-like
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce")
    df = df.dropna(subset=["community_area"])
    df["community_area"] = df["community_area"].astype(int)

    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)

    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))

    def norm_to_grid(v: float, vmin: float, vmax: float) -> int:
        if vmax == vmin:
            return 0
        scaled = (v - vmin) / (vmax - vmin) * (grid_size - 1)
        # Use round to better spread points across the grid
        gi = int(round(float(scaled)))
        return max(0, min(grid_size - 1, gi))

    zone_xy: Dict[int, Tuple[int, int]] = {}
    for zone, la, lo in zip(df["community_area"].to_list(), lat, lon):
        x = norm_to_grid(lo, lon_min, lon_max)
        y = norm_to_grid(la, lat_min, lat_max)
        zone_xy[int(zone)] = (x, y)

    return zone_xy


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Compute Manhattan distance between two grid points."""
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _fairness_allows(
    unit_id: Any,
    zone_id: int,
    risk_score: float,
    fairness_state: MutableMapping[str, Any],
    fairness_cfg: Mapping[str, Any],
) -> bool:
    """
    Enforce:
    - Do not assign the same zone > max_consecutive times in a row for the SAME unit,
      unless risk_score >= high_risk_threshold.

    Expected shapes:
    - fairness_state: dict with optional key "per_unit" mapping unit_id -> {"last_zone": int|None, "consecutive": int}
      If missing, it will be treated as empty state.
    - fairness_cfg keys:
      - max_consecutive (default 2)
      - high_risk_threshold (default: +inf i.e., no exception unless configured)
    """
    max_consecutive = int(fairness_cfg.get("max_consecutive", 2))
    high_risk_threshold = float(fairness_cfg.get("high_risk_threshold", float("inf")))

    if risk_score >= high_risk_threshold:
        return True

    per_unit = fairness_state.get("per_unit", {})
    st = per_unit.get(unit_id, {})
    last_zone = st.get("last_zone", None)
    consecutive = int(st.get("consecutive", 0))

    # If assigning same zone again would exceed max_consecutive, block.
    if last_zone == zone_id and consecutive >= max_consecutive:
        return False

    return True


def _fairness_update(
    unit_id: Any,
    assigned_zone: int,
    fairness_state: MutableMapping[str, Any],
) -> None:
    """Update per-unit fairness state after assignment."""
    per_unit = fairness_state.setdefault("per_unit", {})
    st = per_unit.get(unit_id, {"last_zone": None, "consecutive": 0})
    if st.get("last_zone", None) == assigned_zone:
        st["consecutive"] = int(st.get("consecutive", 0)) + 1
    else:
        st["last_zone"] = assigned_zone
        st["consecutive"] = 1
    per_unit[unit_id] = st


def allocate_patrols_constrained_greedy(
    zones: Iterable[int],
    risk_scores: Mapping[int, float],
    unit_positions: MutableMapping[Any, Tuple[int, int]],
    zone_xy: Mapping[int, Tuple[int, int]],
    fairness_state: MutableMapping[str, Any],
    fairness_cfg: Mapping[str, Any],
) -> Tuple[Dict[Any, int], int]:
    """
    Allocate patrol units to zones using constrained greedy.

    - Rank zones by descending risk_scores
    - Assign each patrol unit to one zone (without replacement where possible)
    - Objective per unit-zone:
        score = risk_score - alpha * manhattan_distance(unit_pos, zone_pos)
    - Apply fairness constraint BEFORE assignment
    - Update unit_positions after assignment

    Args:
        zones: Iterable of zone IDs
        risk_scores: dict-like mapping zone_id -> risk score (higher is riskier)
        unit_positions: dict mapping unit_id -> (x,y) grid coords (updated in-place)
        zone_xy: dict mapping zone_id -> (x,y) grid coords
        fairness_state: mutable state dict tracking consecutive assignments
        fairness_cfg: config dict containing:
            - alpha (float, default 1.0)
            - max_consecutive (int, default 2)
            - high_risk_threshold (float, default +inf)

    Returns:
        assignments: dict unit_id -> zone_id
        total_distance: sum of Manhattan distances traveled by all units
    """
    alpha = float(fairness_cfg.get("alpha", 1.0))

    zones_list = [int(z) for z in zones]
    # Only consider zones that have coordinates; otherwise distance is undefined.
    zones_list = [z for z in zones_list if z in zone_xy]

    # Rank zones by descending risk
    ranked = sorted(zones_list, key=lambda z: float(risk_scores.get(z, 0.0)), reverse=True)

    # Track which zones are already used (so each unit gets a distinct zone when possible)
    remaining = set(ranked)
    assignments: Dict[Any, int] = {}
    total_distance = 0

    for unit_id, unit_xy in unit_positions.items():
        best_zone = None
        best_obj = -float("inf")
        best_dist = 0

        # Iterate zones in ranked order (greedy)
        for zone_id in ranked:
            if zone_id not in remaining:
                continue

            risk = float(risk_scores.get(zone_id, 0.0))
            if not _fairness_allows(unit_id, zone_id, risk, fairness_state, fairness_cfg):
                continue

            d = manhattan_distance(unit_xy, zone_xy[zone_id])
            obj = risk - alpha * d

            if obj > best_obj:
                best_obj = obj
                best_zone = zone_id
                best_dist = d

        # If no zone passes fairness (or no remaining), fall back to highest-risk remaining (ignoring fairness)
        if best_zone is None and remaining:
            for zone_id in ranked:
                if zone_id in remaining:
                    best_zone = zone_id
                    best_dist = manhattan_distance(unit_xy, zone_xy[zone_id])
                    break

        if best_zone is None:
            # No assignable zones
            continue

        assignments[unit_id] = best_zone
        total_distance += int(best_dist)

        # Update positions and fairness
        unit_positions[unit_id] = zone_xy[best_zone]
        _fairness_update(unit_id, best_zone, fairness_state)

        if best_zone in remaining:
            remaining.remove(best_zone)

    return assignments, total_distance
