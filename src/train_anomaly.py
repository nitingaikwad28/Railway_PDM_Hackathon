"""
Train per-asset-type anomaly detectors.

Model choice: scikit-learn IsolationForest, fit only on the "healthy"
portion of each asset's life (first 60% of its run-to-failure trajectory).
IsolationForest is used because:
  - No labeled fault data is required (real fleets rarely have enough
    labeled failures) -- it learns what "normal" multi-sensor behavior
    looks like and flags deviations.
  - Inference is O(n_trees * depth) per sample -- microsecond latency,
    same production-latency reasoning as the RUL model.

The trained artifact bundles: the forest, a decision-score threshold
(1st percentile of healthy validation scores), and per-sensor baseline
mean/std used by the dashboard to show *which* channel is deviating.

Run: python -m src.train_anomaly
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data.simulate_telemetry import ASSET_TYPES, load_or_generate_training_dataset
from src.features import compute_rolling_features, feature_columns, sensor_cols_for

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "serving", "model_store")
os.makedirs(MODEL_DIR, exist_ok=True)

HEALTHY_LIFE_FRACTION = 0.6


def train_one_asset_type(df: pd.DataFrame, asset_type: str) -> dict:
    t0 = time.time()
    sub = df[df["asset_type"] == asset_type].copy()
    sub = compute_rolling_features(sub, asset_type)

    life_frac = sub["cycle"] / sub.groupby("asset_id")["cycle"].transform("max").clip(lower=1)
    healthy = sub[life_frac <= HEALTHY_LIFE_FRACTION]

    raw_cols = sensor_cols_for(asset_type)
    roll_cols = feature_columns(asset_type)
    feat_cols = raw_cols + roll_cols

    X_healthy = healthy[feat_cols].values
    X_all = sub[feat_cols].values

    forest = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination="auto",
        n_jobs=-1,
        random_state=42,
    )
    forest.fit(X_healthy)

    healthy_scores = forest.score_samples(X_healthy)
    threshold = float(np.percentile(healthy_scores, 1))

    baseline = {
        col: {"mean": float(healthy[col].mean()), "std": float(healthy[col].std() or 1e-6)}
        for col in raw_cols
    }

    # Fast real-time gate: mean squared z-score across raw sensor channels
    # (diagonal-covariance Hotelling-T^2-style statistic). This is what the
    # serving hot path uses per event -- O(n_sensors) arithmetic, no tree
    # traversal -- so it can run inside a strict per-message latency budget.
    # IsolationForest above is reserved for periodic *batched* deep scans
    # (see serving/inference.py), where its fixed per-call overhead is
    # amortized across many assets at once instead of paid per event.
    z2 = np.zeros(len(healthy))
    for col in raw_cols:
        b = baseline[col]
        z2 += ((healthy[col].values - b["mean"]) / b["std"]) ** 2
    z2 /= len(raw_cols)
    fast_zscore_threshold = float(np.percentile(z2, 99))

    model_path = os.path.join(MODEL_DIR, f"anomaly_{asset_type}.joblib")
    joblib.dump(forest, model_path)

    all_scores = forest.score_samples(X_all)
    flagged_rate = float(np.mean(all_scores < threshold))

    meta = {
        "asset_type": asset_type,
        "feature_columns": feat_cols,
        "threshold": threshold,
        "baseline": baseline,
        "fast_zscore_threshold": fast_zscore_threshold,
        "train_seconds": round(time.time() - t0, 2),
        "healthy_rows": int(len(X_healthy)),
        "flagged_rate_full_life": round(flagged_rate, 4),
    }
    print(f"[{asset_type}] anomaly model trained in {meta['train_seconds']}s | "
          f"threshold={threshold:.4f} | flag-rate over full life={flagged_rate:.2%}")
    return meta


def main() -> None:
    print("Loading run-to-failure training dataset (data/training_dataset.parquet)...")
    df = load_or_generate_training_dataset()
    all_meta = {}
    for asset_type in ASSET_TYPES:
        all_meta[asset_type] = train_one_asset_type(df, asset_type)

    with open(os.path.join(MODEL_DIR, "anomaly_metadata.json"), "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Saved anomaly models + metadata -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
