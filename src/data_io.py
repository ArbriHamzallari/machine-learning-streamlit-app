"""Data input/output utilities.

STRICT rules per course project:
- Dataset path: data/chicago_crimes_2025.csv
- Required columns (lowercase):
  date, primary_type, arrest, domestic, community_area, latitude, longitude, year
"""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS: list[str] = [
    "date",
    "primary_type",
    "arrest",
    "domestic",
    "community_area",
    "latitude",
    "longitude",
    "year",
]


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load raw CSV using pandas.read_csv (do NOT rename columns here)."""
    return pd.read_csv(csv_path)


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series to boolean.

    Accepts: True/False, "true"/"false" (case-insensitive), 0/1 (numeric or string).
    Any other values become <NA>.
    """
    # Fast-path: already boolean dtype
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.astype("boolean")

    def to_bool(x: Any) -> Any:
        if pd.isna(x):
            return pd.NA
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            if x == 0:
                return False
            if x == 1:
                return True
            return pd.NA
        if isinstance(x, str):
            v = x.strip().lower()
            if v in {"true", "t", "1"}:
                return True
            if v in {"false", "f", "0"}:
                return False
            return pd.NA
        return pd.NA

    return s.map(to_bool).astype("boolean")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset to match required schema.

    Rules:
    - Convert all column names to lowercase
    - Strip whitespace from column names
    - Parse `date` using pandas.to_datetime(errors="coerce")
    - Drop rows where date is null OR community_area is null
    - Convert community_area to integer; drop rows where conversion fails
    - Convert arrest and domestic to boolean; accept True/False, true/false, 0/1
      (drop rows where conversion fails)
    - Drop rows where latitude OR longitude is null
    - Return ONLY required columns (in REQUIRED_COLUMNS order)
    """
    out = df.copy()

    # Normalize column names
    out.columns = [str(c).strip().lower() for c in out.columns]

    # Ensure required columns exist before selecting them
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse date
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Drop rows with missing date/community_area
    out = out.dropna(subset=["date", "community_area"])

    # community_area -> integer (drop conversion failures)
    out["community_area"] = pd.to_numeric(out["community_area"], errors="coerce")
    out = out.dropna(subset=["community_area"])
    # At this point values are numeric; cast to int (safe after dropna)
    out["community_area"] = out["community_area"].astype(int)

    # arrest/domestic -> boolean (drop conversion failures)
    out["arrest"] = _coerce_bool_series(out["arrest"])
    out["domestic"] = _coerce_bool_series(out["domestic"])
    out = out.dropna(subset=["arrest", "domestic"])

    # Require latitude/longitude present
    out = out.dropna(subset=["latitude", "longitude"])

    # Return only required columns (in a stable order)
    out = out.loc[:, REQUIRED_COLUMNS]
    return out


def filter_year(df: pd.DataFrame, year: int = 2025) -> pd.DataFrame:
    """Keep only rows where df['year'] == year."""
    return df[df["year"] == year].copy()


if __name__ == "__main__":
    CSV_PATH = "data/chicago_crimes_2025.csv"

    raw = load_csv(CSV_PATH)
    cleaned = clean_df(raw)
    filtered = filter_year(cleaned, year=2025)

    total_rows = len(filtered)
    min_date = filtered["date"].min() if total_rows else None
    max_date = filtered["date"].max() if total_rows else None
    unique_zones = filtered["community_area"].nunique() if total_rows else 0

    print(f"total rows: {total_rows}")
    print(f"min(date): {min_date}")
    print(f"max(date): {max_date}")
    print(f"unique community_area: {unique_zones}")
