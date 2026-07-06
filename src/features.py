"""
Feature engineering shared by training and the realtime serving path.

Both sides must compute IDENTICAL features or the model will see
train/serve skew. Training uses vectorized pandas rolling ops over the
historical dataset; serving uses an O(1)-per-event incremental buffer
(no re-scan of history) so per-message latency stays flat regardless of
how long an asset has been streaming -- this is what keeps inference
inside a tight production latency budget.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd

WINDOW = 20  # cycles

from data.simulate_telemetry import SENSOR_CHANNELS  # noqa: E402


def sensor_cols_for(asset_type: str) -> list[str]:
    return [c for c in SENSOR_CHANNELS[asset_type] if c != "brake_cycles"]


def _feature_names(sensor_cols: Iterable[str]) -> list[str]:
    names = []
    for s in sensor_cols:
        names += [f"{s}_roll_mean", f"{s}_roll_std", f"{s}_roll_slope", f"{s}_delta"]
    return names


def compute_rolling_features(df: pd.DataFrame, asset_type: str, window: int = WINDOW) -> pd.DataFrame:
    """Batch feature engineering for training. Expects df sorted by
    (asset_id, cycle) and containing only rows for `asset_type`."""
    sensor_cols = sensor_cols_for(asset_type)
    df = df.sort_values(["asset_id", "cycle"]).copy()
    grouped = df.groupby("asset_id", sort=False)

    for s in sensor_cols:
        roll = grouped[s].rolling(window=window, min_periods=1)
        df[f"{s}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{s}_roll_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        df[f"{s}_delta"] = grouped[s].diff().fillna(0.0)

        def _slope(x: pd.Series) -> pd.Series:
            idx = np.arange(len(x))

            def slope_fn(window_vals: np.ndarray) -> float:
                n = len(window_vals)
                if n < 2:
                    return 0.0
                xs = np.arange(n)
                xs_mean = xs.mean()
                ys_mean = window_vals.mean()
                denom = ((xs - xs_mean) ** 2).sum()
                if denom == 0:
                    return 0.0
                return float(((xs - xs_mean) * (window_vals - ys_mean)).sum() / denom)

            return x.rolling(window=window, min_periods=2).apply(slope_fn, raw=True).fillna(0.0)

        df[f"{s}_roll_slope"] = grouped[s].apply(_slope).reset_index(level=0, drop=True)

    return df


class _WelfordRoll:
    """Fixed-window incremental mean/std/slope tracker, O(1) per update."""

    __slots__ = ("window", "buf", "sum", "sumsq")

    def __init__(self, window: int):
        self.window = window
        self.buf: deque[float] = deque(maxlen=window)
        self.sum = 0.0
        self.sumsq = 0.0

    def push(self, value: float) -> None:
        if len(self.buf) == self.buf.maxlen:
            old = self.buf[0]
            self.sum -= old
            self.sumsq -= old * old
        self.buf.append(value)
        self.sum += value
        self.sumsq += value * value

    @property
    def mean(self) -> float:
        n = len(self.buf)
        return self.sum / n if n else 0.0

    @property
    def std(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        var = max(self.sumsq / n - self.mean ** 2, 0.0)
        return float(np.sqrt(var))

    @property
    def slope(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        ys = np.fromiter(self.buf, dtype=float, count=n)
        xs = np.arange(n)
        xs_mean = xs.mean()
        denom = ((xs - xs_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((xs - xs_mean) * (ys - ys.mean())).sum() / denom)


class RollingFeatureBuffer:
    """Per-asset incremental feature state for the realtime serving path.

    Keeps a bounded window (`WINDOW` samples) per sensor per asset, updated
    in O(window) worst case / O(1) amortized, so feature extraction for an
    incoming telemetry event never rescans the asset's full history.
    """

    def __init__(self, window: int = WINDOW):
        self.window = window
        self._trackers: dict[str, dict[str, _WelfordRoll]] = defaultdict(dict)
        self._last_value: dict[str, dict[str, float]] = defaultdict(dict)

    def update(self, asset_id: str, asset_type: str, reading: dict) -> dict:
        """Push one telemetry reading and return the feature row (dict) ready
        for model inference."""
        sensor_cols = sensor_cols_for(asset_type)
        trackers = self._trackers[asset_id]
        last_values = self._last_value[asset_id]
        features: dict[str, float] = {}

        for s in sensor_cols:
            value = float(reading[s])
            if s not in trackers:
                trackers[s] = _WelfordRoll(self.window)
            tracker = trackers[s]
            prev = last_values.get(s, value)
            tracker.push(value)
            last_values[s] = value

            features[f"{s}_roll_mean"] = tracker.mean
            features[f"{s}_roll_std"] = tracker.std
            features[f"{s}_roll_slope"] = tracker.slope
            features[f"{s}_delta"] = value - prev

        return features

    def reset_asset(self, asset_id: str) -> None:
        self._trackers.pop(asset_id, None)
        self._last_value.pop(asset_id, None)


def feature_columns(asset_type: str) -> list[str]:
    return _feature_names(sensor_cols_for(asset_type))
