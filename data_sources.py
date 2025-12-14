from __future__ import annotations

"""
Centralized data loading for the app.

Run `python build_artifacts.py` once to bundle the CSV inputs into
`data_artifacts.joblib`. At runtime the loaders will prefer the
artifact bundle and only fall back to CSVs if the bundle is missing.
"""

from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd


DATA_ARTIFACT_PATH = Path("data_artifacts.joblib")

# Source CSVs that can be bundled into the artifact.
CSV_PATHS: Dict[str, Path] = {
    "voting_data_full": Path("voting_data_full.csv"),
    "legislator_votes": Path("final_legislator_votes_merged (1).csv"),
    "member_summary_pct_by_bucket": Path("member_summary_pct_by_bucket.csv"),
}

_CACHE: Dict[str, pd.DataFrame] = {}
_ARTIFACT_DATA: Optional[Dict[str, pd.DataFrame]] = None


def _load_artifact_data() -> Dict[str, pd.DataFrame]:
    """
    Load the joblib bundle once and keep it in memory.
    """
    global _ARTIFACT_DATA
    if _ARTIFACT_DATA is not None:
        return _ARTIFACT_DATA

    if not DATA_ARTIFACT_PATH.exists():
        _ARTIFACT_DATA = {}
        return _ARTIFACT_DATA

    payload = joblib.load(DATA_ARTIFACT_PATH)
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], dict):
            _ARTIFACT_DATA = payload["data"]
        else:
            _ARTIFACT_DATA = payload  # assume raw dict of dataframes
    else:
        _ARTIFACT_DATA = {}
    return _ARTIFACT_DATA


def _load_dataset(name: str, csv_path: Path) -> pd.DataFrame:
    """
    Fetch a dataset, preferring the artifact bundle and falling back to CSV.
    """
    if name in _CACHE:
        return _CACHE[name]

    artifact_data = _load_artifact_data()
    if name in artifact_data:
        _CACHE[name] = artifact_data[name]
        return _CACHE[name]

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        _CACHE[name] = df
        return df

    raise FileNotFoundError(
        f"Dataset '{name}' not found. Generate artifacts with "
        f"`python build_artifacts.py` or provide {csv_path}."
    )


def get_voting_data() -> pd.DataFrame:
    return _load_dataset("voting_data_full", CSV_PATHS["voting_data_full"])


def get_legislator_votes() -> pd.DataFrame:
    return _load_dataset("legislator_votes", CSV_PATHS["legislator_votes"])


def get_member_summary() -> pd.DataFrame:
    return _load_dataset(
        "member_summary_pct_by_bucket",
        CSV_PATHS["member_summary_pct_by_bucket"],
    )


def using_artifacts() -> bool:
    """
    Quick helper for debugging/logging.
    """
    return DATA_ARTIFACT_PATH.exists()
