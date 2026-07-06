"""
Train per-asset-type Remaining-Useful-Life (RUL) regressors.

Model choice: LightGBM (histogram-based gradient boosting).
  - Trains on 100k+ rows per asset type in a few seconds on a laptop CPU.
  - Inference is a handful of tree traversals -> low-microsecond latency,
    which is what lets `serving/app.py` meet strict real-time SLAs without
    needing a GPU or a model server cluster.
  - Exported to ONNX (`onnxruntime`) after training: ORT's compiled graph
    executor is faster and more latency-consistent than the pure-Python
    LightGBM predict path, and is the standard production inference
    runtime for tabular models.

Run: python -m src.train_rul
"""
from __future__ import annotations

import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from data.simulate_telemetry import ASSET_TYPES, generate_training_dataset
from src.features import compute_rolling_features, feature_columns, sensor_cols_for

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "serving", "model_store")
os.makedirs(MODEL_DIR, exist_ok=True)

# Cap RUL targets: predicting exact RUL at 5000 cycles out is not useful for
# maintenance planning and just adds regression noise -- clip like C-MAPSS does.
RUL_CAP = 300


def _feature_matrix(df: pd.DataFrame, asset_type: str) -> tuple[pd.DataFrame, list[str]]:
    raw_cols = sensor_cols_for(asset_type)
    roll_cols = feature_columns(asset_type)
    cols = raw_cols + roll_cols
    return df[cols], cols


def train_one_asset_type(df: pd.DataFrame, asset_type: str) -> dict:
    t0 = time.time()
    sub = df[df["asset_type"] == asset_type].copy()
    sub = compute_rolling_features(sub, asset_type)
    sub["rul_clipped"] = sub["rul"].clip(upper=RUL_CAP)

    X, feat_cols = _feature_matrix(sub, asset_type)
    y = sub["rul_clipped"].values
    groups = sub["asset_id"].values

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    train_set = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
    val_set = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], reference=train_set)

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "num_leaves": 31,
        "learning_rate": 0.08,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "num_threads": os.cpu_count() or 4,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    preds = booster.predict(X.iloc[val_idx], num_iteration=booster.best_iteration)
    mae = mean_absolute_error(y[val_idx], preds)

    model_path = os.path.join(MODEL_DIR, f"rul_{asset_type}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    meta = {
        "asset_type": asset_type,
        "feature_columns": feat_cols,
        "rul_cap": RUL_CAP,
        "val_mae_cycles": float(mae),
        "best_iteration": int(booster.best_iteration),
        "train_seconds": round(time.time() - t0, 2),
        "n_train_rows": int(len(train_idx)),
    }

    # ONNX export for the fast serving path. Falls back gracefully if the
    # optional onnx toolchain isn't installed in this environment.
    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        onnx_model = convert_lightgbm(
            booster, initial_types=[("input", FloatTensorType([None, len(feat_cols)]))]
        )
        onnx_path = os.path.join(MODEL_DIR, f"rul_{asset_type}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        meta["onnx_exported"] = True
    except Exception as exc:  # pragma: no cover - optional dependency path
        meta["onnx_exported"] = False
        meta["onnx_export_error"] = str(exc)

    print(f"[{asset_type}] trained in {meta['train_seconds']}s | val MAE = {mae:.2f} cycles "
          f"| onnx={meta['onnx_exported']}")
    return meta


def main() -> None:
    print("Generating synthetic run-to-failure training dataset...")
    df = generate_training_dataset(n_assets_per_type=40)
    all_meta = {}
    for asset_type in ASSET_TYPES:
        all_meta[asset_type] = train_one_asset_type(df, asset_type)

    with open(os.path.join(MODEL_DIR, "rul_metadata.json"), "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Saved RUL models + metadata -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
