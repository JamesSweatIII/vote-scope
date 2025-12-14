from __future__ import annotations

"""
Bundle all CSV inputs into a single joblib artifact for faster, CSV-free runtime.

Run once:
    python build_artifacts.py
"""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from data_sources import CSV_PATHS, DATA_ARTIFACT_PATH


def build() -> None:
    artifacts: Dict[str, pd.DataFrame] = {}
    metadata: Dict[str, Dict[str, object]] = {"sources": {}, "rows": {}}

    for name, csv_path in CSV_PATHS.items():
        if not csv_path.exists():
            print(f"[skip] {name}: missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        artifacts[name] = df
        metadata["sources"][name] = str(csv_path.resolve())
        metadata["rows"][name] = len(df)
        print(f"[load] {name}: {len(df)} rows from {csv_path}")

    payload = {"data": artifacts, "metadata": metadata}
    DATA_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, DATA_ARTIFACT_PATH)
    print(f"[done] wrote artifact bundle to {DATA_ARTIFACT_PATH}")

    if len(artifacts) != len(CSV_PATHS):
        missing = set(CSV_PATHS) - set(artifacts)
        print(f"Note: missing datasets not bundled: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    build()
