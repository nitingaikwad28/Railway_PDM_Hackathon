"""
Model evaluation & benchmarking.

Produces the numbers required by the hackathon deliverables:

  * RUL accuracy benchmarks -- MAE / RMSE / R^2 overall, per asset type,
    and broken down by how close the asset is to failure (accuracy near
    end-of-life matters most for planning).
  * Anomaly-detector operating metrics -- **detection lead time** (how many
    operating cycles BEFORE end-of-life we first raise a sustained alarm)
    and **false-alarm rate** (fraction of healthy-life readings wrongly
    flagged), plus precision / recall / F1 on a "failure-imminent" label.

Method: we generate a FRESH held-out fleet of run-to-failure assets with a
different random seed from training, then replay each asset cycle-by-cycle
through the exact production `InferenceEngine.process_event` path -- so these
metrics reflect the real serving behaviour, not an idealized offline model.

Run:  python -m src.evaluate
Output: printed tables + serving/model_store/evaluation_report.json
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from data.simulate_telemetry import ASSET_TYPES, generate_training_dataset
from serving.inference import InferenceEngine
from src.features import sensor_cols_for

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "serving", "model_store")

# --- evaluation config -------------------------------------------------------
TEST_SEED = 2024              # different from training seed (42) -> unseen assets
N_TEST_ASSETS_PER_TYPE = 12
IMMINENT_FRACTION = 0.10      # last 10% of life = "failure imminent" (positive class)
DEBOUNCE = 3                  # need this many consecutive flags to declare detection
RUL_BANDS = [(0, 50), (50, 150), (150, 300), (300, 10_000)]


def _make_test_fleet() -> pd.DataFrame:
    """Held-out run-to-failure assets the models have never seen."""
    df = generate_training_dataset(n_assets_per_type=N_TEST_ASSETS_PER_TYPE, seed=TEST_SEED)
    return df


def _replay_asset(engine: InferenceEngine, asset_df: pd.DataFrame) -> pd.DataFrame:
    """Replay one asset's whole life through the production scoring path.
    Returns a per-cycle frame with predictions aligned to ground truth."""
    asset_df = asset_df.sort_values("cycle")
    asset_type = asset_df["asset_type"].iloc[0]
    raw_cols = sensor_cols_for(asset_type)

    rows = []
    for _, r in asset_df.iterrows():
        reading = {
            "asset_id": r["asset_id"],
            "asset_type": asset_type,
            "train_id": "eval",
            "cycle": int(r["cycle"]),
            "timestamp": 0.0,
        }
        for c in raw_cols:
            reading[c] = float(r[c])
        res = engine.process_event(reading)
        rows.append({
            "cycle": int(r["cycle"]),
            "true_rul": int(r["rul"]),
            "pred_rul": res.rul_pred_cycles,
            "is_anomaly": res.is_anomaly,
            "anomaly_score": res.anomaly_score,
        })
    out = pd.DataFrame(rows)
    # reset the per-asset rolling buffer so assets don't bleed into each other
    engine.buffers.reset_asset(asset_df["asset_id"].iloc[0])
    return out


def _rul_metrics(pred: np.ndarray, true: np.ndarray, cap: float) -> dict:
    # Compare on the capped target the model was trained to predict, so the
    # benchmark is not dominated by the long, deliberately-flat healthy tail.
    true_capped = np.minimum(true, cap)
    mae = float(mean_absolute_error(true_capped, pred))
    rmse = float(np.sqrt(np.mean((true_capped - pred) ** 2)))
    r2 = float(r2_score(true_capped, pred)) if len(np.unique(true_capped)) > 1 else float("nan")
    return {"mae_cycles": round(mae, 2), "rmse_cycles": round(rmse, 2),
            "r2": round(r2, 4), "n": int(len(pred))}


def _rul_by_band(pred: np.ndarray, true_capped: np.ndarray) -> dict:
    out = {}
    for lo, hi in RUL_BANDS:
        m = (true_capped >= lo) & (true_capped < hi)
        if m.sum() == 0:
            continue
        out[f"{lo}-{hi if hi < 10_000 else 'cap'}"] = {
            "mae_cycles": round(float(mean_absolute_error(true_capped[m], pred[m])), 2),
            "n": int(m.sum()),
        }
    return out


def _anomaly_metrics_for_asset(replay: pd.DataFrame, eol: int) -> dict:
    """Compute per-asset lead time and false-alarm stats from a replayed life."""
    imminent_start = int((1 - IMMINENT_FRACTION) * eol)
    healthy_end = int(0.6 * eol)  # first 60% considered genuinely healthy

    flags = replay["is_anomaly"].values.astype(bool)
    cycles = replay["cycle"].values

    # False alarms: flags raised during the healthy early-life window.
    healthy_mask = cycles < healthy_end
    n_healthy = int(healthy_mask.sum())
    n_false_alarm = int(flags[healthy_mask].sum())

    # Detection: first cycle with DEBOUNCE consecutive flags anywhere in life.
    detect_cycle = None
    run = 0
    for c, f in zip(cycles, flags):
        run = run + 1 if f else 0
        if run >= DEBOUNCE:
            detect_cycle = int(c) - (DEBOUNCE - 1)
            break

    detected_before_eol = detect_cycle is not None and detect_cycle < eol
    lead_time = (eol - detect_cycle) if detected_before_eol else None
    # Did we catch it before the imminent-failure window opened? (proactive win)
    detected_before_imminent = detect_cycle is not None and detect_cycle <= imminent_start

    # Point-wise confusion. Positive class = "degrading" (past the healthy
    # window the model was trained on, life fraction >= 0.6), which is the
    # detector's actual decision boundary -- healthy vs. not-healthy. Scoring
    # against the last-10% window instead would unfairly count every correct
    # early warning as a false positive.
    degrading_label = cycles >= healthy_end
    tp = int((flags & degrading_label).sum())
    fp = int((flags & ~degrading_label).sum())
    fn = int((~flags & degrading_label).sum())
    tn = int((~flags & ~degrading_label).sum())

    return {
        "eol": eol,
        "n_healthy": n_healthy,
        "n_false_alarm": n_false_alarm,
        "detect_cycle": detect_cycle,
        "detected_before_eol": detected_before_eol,
        "detected_before_imminent": detected_before_imminent,
        "lead_time_cycles": lead_time,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate() -> dict:
    t0 = time.time()
    print("Generating held-out test fleet (unseen assets)...")
    fleet = _make_test_fleet()

    engine = InferenceEngine()
    with open(os.path.join(MODEL_DIR, "rul_metadata.json")) as f:
        rul_meta = json.load(f)

    report: dict = {
        "config": {
            "test_seed": TEST_SEED,
            "n_test_assets_per_type": N_TEST_ASSETS_PER_TYPE,
            "imminent_fraction": IMMINENT_FRACTION,
            "debounce": DEBOUNCE,
        },
        "rul_benchmarks": {},
        "anomaly_metrics": {},
    }

    overall_pred, overall_true, overall_cap = [], [], []
    overall_anom: list[dict] = []

    for asset_type in ASSET_TYPES:
        cap = rul_meta[asset_type]["rul_cap"]
        sub = fleet[fleet["asset_type"] == asset_type]

        type_pred, type_true = [], []
        type_anom: list[dict] = []

        for asset_id, asset_df in sub.groupby("asset_id"):
            replay = _replay_asset(engine, asset_df)
            type_pred.append(replay["pred_rul"].values)
            type_true.append(replay["true_rul"].values)
            eol = int(asset_df["cycle"].max()) + 1
            type_anom.append(_anomaly_metrics_for_asset(replay, eol))

        pred = np.concatenate(type_pred)
        true = np.concatenate(type_true)
        true_capped = np.minimum(true, cap)

        report["rul_benchmarks"][asset_type] = {
            **_rul_metrics(pred, true, cap),
            "by_rul_band": _rul_by_band(pred, true_capped),
        }
        report["anomaly_metrics"][asset_type] = _aggregate_anomaly(type_anom)

        overall_pred.append(pred)
        overall_true.append(true)
        overall_cap.append(np.full(len(pred), cap))
        overall_anom.extend(type_anom)

    # Overall RUL (per-type caps applied element-wise before concat)
    op = np.concatenate(overall_pred)
    ot = np.concatenate(overall_true)
    oc = np.concatenate(overall_cap)
    ot_capped = np.minimum(ot, oc)
    report["rul_benchmarks"]["overall"] = {
        "mae_cycles": round(float(mean_absolute_error(ot_capped, op)), 2),
        "rmse_cycles": round(float(np.sqrt(np.mean((ot_capped - op) ** 2))), 2),
        "r2": round(float(r2_score(ot_capped, op)), 4),
        "n": int(len(op)),
    }
    report["anomaly_metrics"]["overall"] = _aggregate_anomaly(overall_anom)
    report["eval_seconds"] = round(time.time() - t0, 2)

    out_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    print(f"\nSaved evaluation report -> {out_path}  ({report['eval_seconds']}s)")
    return report


def _aggregate_anomaly(per_asset: list[dict]) -> dict:
    n_assets = len(per_asset)
    lead_times = [a["lead_time_cycles"] for a in per_asset if a["lead_time_cycles"] is not None]
    n_detected = sum(1 for a in per_asset if a["detected_before_eol"])
    n_proactive = sum(1 for a in per_asset if a["detected_before_imminent"])

    tot_healthy = sum(a["n_healthy"] for a in per_asset)
    tot_false = sum(a["n_false_alarm"] for a in per_asset)

    tp = sum(a["tp"] for a in per_asset)
    fp = sum(a["fp"] for a in per_asset)
    fn = sum(a["fn"] for a in per_asset)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision) and not np.isnan(recall) else float("nan"))

    return {
        "n_assets": n_assets,
        "detection_rate": round(n_detected / n_assets, 4) if n_assets else None,
        "proactive_rate": round(n_proactive / n_assets, 4) if n_assets else None,
        "mean_lead_time_cycles": round(float(np.mean(lead_times)), 1) if lead_times else None,
        "median_lead_time_cycles": round(float(np.median(lead_times)), 1) if lead_times else None,
        "false_alarm_rate": round(tot_false / tot_healthy, 4) if tot_healthy else None,
        "precision": round(precision, 4) if not np.isnan(precision) else None,
        "recall": round(recall, 4) if not np.isnan(recall) else None,
        "f1": round(f1, 4) if not np.isnan(f1) else None,
    }


def _print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("RUL ACCURACY BENCHMARKS  (error in operating cycles; lower is better)")
    print("=" * 70)
    print(f"{'asset_type':<18}{'MAE':>8}{'RMSE':>9}{'R2':>8}{'n':>10}")
    for k, v in report["rul_benchmarks"].items():
        print(f"{k:<18}{v['mae_cycles']:>8}{v['rmse_cycles']:>9}{v['r2']:>8}{v['n']:>10}")

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION METRICS")
    print("=" * 70)
    hdr = (f"{'asset_type':<18}{'det.rate':>9}{'lead(med)':>11}"
           f"{'false_alarm':>13}{'precision':>11}{'recall':>9}")
    print(hdr)
    for k, v in report["anomaly_metrics"].items():
        print(f"{k:<18}{_fmt(v['detection_rate']):>9}{_fmt(v['median_lead_time_cycles']):>11}"
              f"{_fmt(v['false_alarm_rate']):>13}{_fmt(v['precision']):>11}{_fmt(v['recall']):>9}")
    print("\nlead(med) = median operating cycles of early warning before end-of-life")
    print("false_alarm = fraction of healthy-life readings wrongly flagged")


def _fmt(x) -> str:
    return "-" if x is None else (f"{x}")


if __name__ == "__main__":
    evaluate()
