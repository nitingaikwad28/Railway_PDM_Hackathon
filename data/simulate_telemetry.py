"""
Synthetic railway telemetry generator.

Mimics live multi-channel sensor streams (vibration, temperature, pressure,
current, operational cycles) for three critical rail asset classes:
bogie bearings, brake units and traction motors.

Two things are produced from the same physical model:

1. A historical run-to-failure dataset (`generate_training_dataset`) used to
   train the anomaly detector and the Remaining-Useful-Life (RUL) regressor.
2. A live event generator (`LiveTelemetrySimulator`) that yields one
   telemetry record at a time, the same shape a real onboard sensor gateway
   would push over MQTT/Kafka, for demoing the realtime dashboard + API.

Degradation model: each asset instance is assigned a random end-of-life
(EOL) in operating cycles. Sensor channels stay flat/noisy while healthy,
then pick up an exponential degradation trend as the asset approaches EOL
(the same "knee curve" shape used in NASA C-MAPSS), so RUL is learnable
from rolling trend/variance features.
"""
from __future__ import annotations

import dataclasses
import os
import time
import uuid
from typing import Iterator, Optional

import numpy as np
import pandas as pd

RNG_SEED = 42

ASSET_TYPES = ("bogie_bearing", "brake_unit", "traction_motor")

SENSOR_CHANNELS = {
    "bogie_bearing": ["vibration_rms", "vibration_kurtosis", "temperature_c", "axle_load_kn"],
    "brake_unit": ["pad_wear_mm", "disc_temperature_c", "brake_pressure_bar", "brake_cycles"],
    "traction_motor": ["winding_temperature_c", "vibration_rms", "current_draw_a", "rpm"],
}

# (baseline_mean, baseline_std, degradation_gain, noise_std, direction)
# direction: +1 sensor rises with degradation, -1 sensor falls with degradation
SENSOR_PROFILES = {
    "bogie_bearing": {
        "vibration_rms": (0.35, 0.03, 2.8, 0.02, +1),
        "vibration_kurtosis": (3.0, 0.15, 4.0, 0.1, +1),
        "temperature_c": (45.0, 2.0, 25.0, 1.0, +1),
        "axle_load_kn": (120.0, 4.0, 0.0, 2.0, 0),  # exogenous, not degradation-linked
    },
    "brake_unit": {
        "pad_wear_mm": (18.0, 0.5, -12.0, 0.15, -1),  # thickness shrinks toward failure
        "disc_temperature_c": (180.0, 10.0, 90.0, 5.0, +1),
        "brake_pressure_bar": (6.0, 0.2, -1.2, 0.08, -1),
        "brake_cycles": (0.0, 0.0, 0.0, 0.0, 0),  # counter, handled separately
    },
    "traction_motor": {
        "winding_temperature_c": (70.0, 3.0, 35.0, 1.5, +1),
        "vibration_rms": (0.3, 0.02, 2.2, 0.02, +1),
        "current_draw_a": (180.0, 6.0, 40.0, 3.0, +1),
        "rpm": (1480.0, 15.0, -60.0, 8.0, -1),
    },
}

EOL_RANGE = {
    "bogie_bearing": (2500, 6000),
    "brake_unit": (1200, 3500),
    "traction_motor": (3000, 7000),
}


def _degradation_curve(cycle: np.ndarray, eol: int) -> np.ndarray:
    """Exponential knee curve: ~0 while healthy, ramps up near end-of-life."""
    frac = np.clip(cycle / eol, 0.0, 1.2)
    return np.expm1(3.5 * np.clip(frac, 0, 1) ** 3) / np.expm1(3.5)


def _simulate_asset_life(asset_type: str, eol: int, rng: np.random.Generator) -> pd.DataFrame:
    cycles = np.arange(eol)
    curve = _degradation_curve(cycles, eol)
    data = {"cycle": cycles}
    for sensor, (mean, std, gain, noise, direction) in SENSOR_PROFILES[asset_type].items():
        if sensor == "brake_cycles":
            data[sensor] = cycles
            continue
        if direction == 0:  # exogenous / independent signal
            data[sensor] = rng.normal(mean, std, size=eol)
            continue
        trend = direction * gain * curve
        noise_series = rng.normal(0, noise, size=eol)
        data[sensor] = mean + trend + noise_series
    df = pd.DataFrame(data)
    df["rul"] = eol - cycles - 1
    df["is_failure_imminent"] = (df["rul"] <= int(0.1 * eol)).astype(int)
    return df


def generate_training_dataset(
    n_assets_per_type: int = 25,
    seed: int = RNG_SEED,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Vectorized-per-asset run-to-failure dataset generation for model training."""
    rng = np.random.default_rng(seed)
    frames = []
    for asset_type in ASSET_TYPES:
        lo, hi = EOL_RANGE[asset_type]
        for i in range(n_assets_per_type):
            eol = int(rng.integers(lo, hi))
            asset_id = f"{asset_type}_{i:03d}"
            df = _simulate_asset_life(asset_type, eol, rng)
            df["asset_id"] = asset_id
            df["asset_type"] = asset_type
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full[
        ["asset_id", "asset_type", "cycle", "rul", "is_failure_imminent"]
        + sorted({c for cols in SENSOR_CHANNELS.values() for c in cols})
    ]
    if output_path:
        full.to_parquet(output_path, index=False)
    return full


# Canonical training-dataset location + parameters. The training and anomaly
# scripts share this single materialized file so the saved data on disk is the
# exact data the models are trained on (no silent regeneration divergence).
_DATA_DIR = os.path.join(os.path.dirname(__file__))
DEFAULT_TRAINING_PARQUET = os.path.join(_DATA_DIR, "training_dataset.parquet")
DEFAULT_TRAIN_ASSETS_PER_TYPE = 40


def load_or_generate_training_dataset(
    n_assets_per_type: int = DEFAULT_TRAIN_ASSETS_PER_TYPE,
    seed: int = RNG_SEED,
    parquet_path: str = DEFAULT_TRAINING_PARQUET,
) -> pd.DataFrame:
    """Return the canonical training dataset, reading the saved parquet if it
    exists and generating + saving it otherwise. Guarantees the dataset is
    present on disk under data/ and that every consumer uses the same rows."""
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return generate_training_dataset(
        n_assets_per_type=n_assets_per_type, seed=seed, output_path=parquet_path
    )


@dataclasses.dataclass
class LiveAssetState:
    asset_id: str
    asset_type: str
    train_id: str
    eol: int
    cycle: int = 0
    fault_injected_at: Optional[int] = None


class LiveTelemetrySimulator:
    """Generator that yields telemetry events one at a time, mimicking a live
    onboard sensor gateway streaming to a rail-side ingestion endpoint.

    Randomly injects sudden-fault anomalies into a fraction of assets so the
    anomaly detector / dashboard has something interesting to catch live.
    """

    def __init__(
        self,
        n_trains: int = 6,
        assets_per_train_per_type: int = 2,
        anomaly_injection_prob: float = 0.0008,
        seed: int = RNG_SEED,
    ):
        self.rng = np.random.default_rng(seed)
        self.anomaly_injection_prob = anomaly_injection_prob
        self.assets: list[LiveAssetState] = []
        for t in range(n_trains):
            train_id = f"train_{t:02d}"
            for asset_type in ASSET_TYPES:
                lo, hi = EOL_RANGE[asset_type]
                for a in range(assets_per_train_per_type):
                    eol = int(self.rng.integers(lo, hi))
                    start_cycle = int(self.rng.integers(0, int(eol * 0.6)))
                    self.assets.append(
                        LiveAssetState(
                            asset_id=f"{train_id}_{asset_type}_{a}",
                            asset_type=asset_type,
                            train_id=train_id,
                            eol=eol,
                            cycle=start_cycle,
                        )
                    )

    def _read_asset(self, state: LiveAssetState) -> dict:
        curve = _degradation_curve(np.array([state.cycle]), state.eol)[0]
        if state.fault_injected_at is not None:
            elapsed = state.cycle - state.fault_injected_at
            curve = min(1.0, curve + 0.6 * np.exp(-elapsed / 40.0) + 0.15)

        record = {
            "timestamp": time.time(),
            "asset_id": state.asset_id,
            "asset_type": state.asset_type,
            "train_id": state.train_id,
            "cycle": state.cycle,
        }
        for sensor, (mean, std, gain, noise, direction) in SENSOR_PROFILES[state.asset_type].items():
            if sensor == "brake_cycles":
                record[sensor] = float(state.cycle)
                continue
            if direction == 0:
                record[sensor] = float(self.rng.normal(mean, std))
                continue
            trend = direction * gain * curve
            record[sensor] = float(mean + trend + self.rng.normal(0, noise))

        record["true_rul"] = max(state.eol - state.cycle - 1, 0)
        return record

    def stream(self) -> Iterator[dict]:
        """Infinite generator yielding one telemetry record per call, round-robin
        across all simulated assets. Suitable for `for event in sim.stream(): ...`
        or wrapping in a WebSocket/Kafka producer loop."""
        while True:
            for state in self.assets:
                if state.fault_injected_at is None and self.rng.random() < self.anomaly_injection_prob:
                    state.fault_injected_at = state.cycle

                yield self._read_asset(state)

                state.cycle += 1
                if state.cycle >= state.eol:
                    # asset replaced/serviced -> reset to a fresh healthy life
                    lo, hi = EOL_RANGE[state.asset_type]
                    state.eol = int(self.rng.integers(lo, hi))
                    state.cycle = 0
                    state.fault_injected_at = None

    def next_batch(self, n: int) -> list[dict]:
        it = self.stream()
        return [next(it) for _ in range(n)]


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "training_dataset.parquet")
    df = generate_training_dataset(n_assets_per_type=25, output_path=out_path)
    print(f"Generated {len(df):,} rows -> {out_path}")
    print(df.groupby("asset_type")["asset_id"].nunique())
