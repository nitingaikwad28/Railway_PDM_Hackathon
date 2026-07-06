"""
Low-latency inference engine shared by the REST/WebSocket API.

Design for production-grade latency:
  - Models are loaded ONCE at process start (no per-request disk I/O).
  - RUL prediction prefers an ONNX Runtime session (compiled graph, no
    Python interpreter overhead per node) over the native LightGBM
    predict path; falls back automatically if ONNX artifacts are absent.
  - Per-asset feature state lives in an in-process `RollingFeatureBuffer`
    (O(1) amortized update), so a request's latency does not grow with
    how long the asset has been streaming.
  - Everything here is pure numpy / C-extension calls -- no network hops,
    no per-request disk access -- keeping p99 latency in the low
    milliseconds, which is what "stringent time expectations" requires
    for a stream that may be pushing hundreds of readings/sec fleet-wide.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np

from data.simulate_telemetry import ASSET_TYPES, SENSOR_PROFILES
from src.features import RollingFeatureBuffer, feature_columns, sensor_cols_for

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_store")

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _ONNX_AVAILABLE = False


@dataclass
class AssetTypeModels:
    asset_type: str
    rul_feature_cols: list[str]
    rul_cap: float
    anomaly_feature_cols: list[str]
    anomaly_threshold: float
    baseline: dict
    fast_zscore_threshold: float
    rul_onnx_session: "ort.InferenceSession | None" = None
    rul_booster: lgb.Booster | None = None
    anomaly_forest: object = None


@dataclass
class PredictionResult:
    asset_id: str
    asset_type: str
    train_id: str
    cycle: int
    rul_pred_cycles: float
    anomaly_score: float
    is_anomaly: bool
    worst_sensor: str | None
    priority_score: float
    inference_latency_ms: float
    timestamp: float


class InferenceEngine:
    """Holds all per-asset-type models in memory and serves predictions."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.models: dict[str, AssetTypeModels] = {}
        self.buffers = RollingFeatureBuffer()
        # asset_type -> {asset_id: feature_row}, refreshed every process_event call,
        # consumed by run_deep_scan() to batch-score IsolationForest periodically
        # instead of per-event (see module docstring for why).
        self._deep_scan_rows: dict[str, dict[str, np.ndarray]] = {}
        self.deep_scores: dict[str, float] = {}
        self._load_all()

    def _load_all(self) -> None:
        with open(os.path.join(self.model_dir, "rul_metadata.json")) as f:
            rul_meta = json.load(f)
        with open(os.path.join(self.model_dir, "anomaly_metadata.json")) as f:
            anomaly_meta = json.load(f)

        for asset_type in ASSET_TYPES:
            rm = rul_meta[asset_type]
            am = anomaly_meta[asset_type]

            onnx_session = None
            onnx_path = os.path.join(self.model_dir, f"rul_{asset_type}.onnx")
            if _ONNX_AVAILABLE and os.path.exists(onnx_path):
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1  # single-row inference: avoid thread overhead
                onnx_session = ort.InferenceSession(onnx_path, sess_options=so,
                                                     providers=["CPUExecutionProvider"])

            booster = lgb.Booster(model_file=os.path.join(self.model_dir, f"rul_{asset_type}.txt"))
            forest = joblib.load(os.path.join(self.model_dir, f"anomaly_{asset_type}.joblib"))

            self.models[asset_type] = AssetTypeModels(
                asset_type=asset_type,
                rul_feature_cols=rm["feature_columns"],
                rul_cap=rm["rul_cap"],
                anomaly_feature_cols=am["feature_columns"],
                anomaly_threshold=am["threshold"],
                baseline=am["baseline"],
                fast_zscore_threshold=am["fast_zscore_threshold"],
                rul_onnx_session=onnx_session,
                rul_booster=booster,
                anomaly_forest=forest,
            )
            self._deep_scan_rows[asset_type] = {}

    def _predict_rul(self, m: AssetTypeModels, X: np.ndarray) -> float:
        if m.rul_onnx_session is not None:
            out = m.rul_onnx_session.run(None, {"input": X.astype(np.float32)})
            return float(np.asarray(out[0]).reshape(-1)[0])
        return float(m.rul_booster.predict(X)[0])

    def process_event(self, reading: dict) -> PredictionResult:
        t0 = time.perf_counter()
        asset_id = reading["asset_id"]
        asset_type = reading["asset_type"]
        m = self.models[asset_type]

        roll_feats = self.buffers.update(asset_id, asset_type, reading)
        raw_cols = sensor_cols_for(asset_type)
        raw_feats = {c: float(reading[c]) for c in raw_cols}
        all_feats = {**raw_feats, **roll_feats}

        X = np.array([[all_feats[c] for c in m.rul_feature_cols]], dtype=np.float32)
        rul_pred = min(self._predict_rul(m, X), m.rul_cap)

        # Fast real-time anomaly gate: O(n_sensors) z-score statistic against
        # the healthy baseline -- no tree traversal, keeps the hot path inside
        # a strict per-event latency budget (see train_anomaly.py docstring).
        z2_sum = 0.0
        worst_z, worst_sensor = 0.0, None
        for c in raw_cols:
            base = m.baseline[c]
            z = (raw_feats[c] - base["mean"]) / base["std"]
            z2_sum += z * z
            if abs(z) > worst_z:
                worst_z, worst_sensor = abs(z), c
        anomaly_score = z2_sum / len(raw_cols)
        is_anomaly = anomaly_score > m.fast_zscore_threshold
        if not is_anomaly:
            worst_sensor = None

        # Cache the row for the periodic batched IsolationForest deep scan
        # (amortizes IsolationForest's ~20ms fixed per-call overhead across
        # the whole fleet instead of paying it on every single event).
        self._deep_scan_rows.setdefault(asset_type, {})[asset_id] = np.array(
            [all_feats[c] for c in m.anomaly_feature_cols], dtype=np.float64
        )

        # Priority: low RUL and/or strong anomaly both push an asset up the
        # maintenance queue. Normalize RUL to [0,1] urgency, blend with the
        # z-score anomaly statistic normalized against its own threshold.
        rul_urgency = 1.0 - min(rul_pred / m.rul_cap, 1.0)
        anomaly_urgency = min(anomaly_score / (m.fast_zscore_threshold * 3.0), 1.0)
        priority_score = 0.6 * rul_urgency + 0.4 * anomaly_urgency

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PredictionResult(
            asset_id=asset_id,
            asset_type=asset_type,
            train_id=reading.get("train_id", "unknown"),
            cycle=int(reading["cycle"]),
            rul_pred_cycles=round(rul_pred, 1),
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=bool(is_anomaly),
            worst_sensor=worst_sensor,
            priority_score=round(priority_score, 4),
            inference_latency_ms=round(latency_ms, 3),
            timestamp=reading.get("timestamp", time.time()),
        )

    def run_deep_scan(self) -> dict[str, float]:
        """Batched IsolationForest sweep across every asset currently buffered,
        one call per asset type. Not on the per-event hot path: IsolationForest
        has ~20ms of fixed overhead per call regardless of batch size, so this
        is only cheap when amortized across many assets at once (see
        train_anomaly.py). Intended to run on a timer (e.g. every 5-10s) from
        the serving app, giving a slower, higher-fidelity multivariate check
        that complements the instant per-event z-score gate.
        """
        updated: dict[str, float] = {}
        for asset_type, rows in self._deep_scan_rows.items():
            if not rows:
                continue
            asset_ids = list(rows.keys())
            X = np.vstack([rows[a] for a in asset_ids])
            scores = self.models[asset_type].anomaly_forest.score_samples(X)
            for asset_id, score in zip(asset_ids, scores):
                updated[asset_id] = float(score)
        self.deep_scores.update(updated)
        return updated
