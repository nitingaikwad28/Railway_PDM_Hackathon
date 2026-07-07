"""
Materialize the representative telemetry datasets to disk under data/.

Running this is optional (the training scripts auto-generate the parquet on
first run), but it's the explicit way to produce the on-disk "representative
sensor dataset" deliverable and some human-readable CSV samples you can open
in Excel / a text editor.

Outputs (all under data/):
  training_dataset.parquet        full run-to-failure training set (all assets)
  training_dataset_sample.csv     readable sample: one full asset life per type
  live_telemetry_sample.csv       a captured window of the live streaming feed

Run:  python -m data.generate_dataset
"""
from __future__ import annotations

import os

import pandas as pd

from data.simulate_telemetry import (
    ASSET_TYPES,
    DEFAULT_TRAIN_ASSETS_PER_TYPE,
    DEFAULT_TRAINING_PARQUET,
    LiveTelemetrySimulator,
    generate_training_dataset,
)

DATA_DIR = os.path.dirname(__file__)
SAMPLE_CSV = os.path.join(DATA_DIR, "training_dataset_sample.csv")
LIVE_CSV = os.path.join(DATA_DIR, "live_telemetry_sample.csv")

N_LIVE_EVENTS = 600


def main() -> None:
    # 1) Full training dataset -> parquet (efficient, this is what models train on)
    print("Generating full run-to-failure training dataset...")
    df = generate_training_dataset(
        n_assets_per_type=DEFAULT_TRAIN_ASSETS_PER_TYPE,
        output_path=DEFAULT_TRAINING_PARQUET,
    )
    size_mb = os.path.getsize(DEFAULT_TRAINING_PARQUET) / 1e6
    print(f"  -> {DEFAULT_TRAINING_PARQUET}  ({len(df):,} rows, {size_mb:.1f} MB)")

    # 2) Human-readable sample: the first asset of each type, full life, as CSV.
    #    (A single asset's whole run-to-failure trajectory is the clearest thing
    #    to eyeball -- you can see each sensor drift as RUL counts down to 0.)
    sample_ids = [f"{atype}_000" for atype in ASSET_TYPES]
    sample = df[df["asset_id"].isin(sample_ids)].copy()
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"  -> {SAMPLE_CSV}  ({len(sample):,} rows, {len(sample_ids)} assets)")

    # 3) Live-stream capture: what the real-time feed pushes, event by event.
    print(f"Capturing {N_LIVE_EVENTS} live telemetry events...")
    sim = LiveTelemetrySimulator(n_trains=3, assets_per_train_per_type=1)
    events = sim.next_batch(N_LIVE_EVENTS)
    live_df = pd.DataFrame(events)
    live_df.to_csv(LIVE_CSV, index=False)
    print(f"  -> {LIVE_CSV}  ({len(live_df):,} rows)")

    print("\nDataset summary (assets per type):")
    print(df.groupby("asset_type")["asset_id"].nunique().to_string())
    print(f"\nSensor columns: {[c for c in df.columns if c not in ('asset_id','asset_type','cycle','rul','is_failure_imminent')]}")


if __name__ == "__main__":
    main()
