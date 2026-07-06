# Railway Predictive Maintenance -- Hackathon MVP

AI-powered predictive maintenance for rail assets: multi-channel anomaly
detection + Remaining-Useful-Life (RUL) estimation, served behind a
low-latency real-time API, with a fleet-level maintenance dashboard and an
MRO/EAM integration stub. Built against the brief in
[`Railway Hackathon Idea.txt`](Railway%20Hackathon%20Idea.txt).

## Architecture

```
data/simulate_telemetry.py   synthetic multi-channel telemetry (train + live stream)
src/features.py              feature engineering shared by training & serving
src/train_rul.py             LightGBM RUL regressor per asset type -> ONNX export
src/train_anomaly.py         IsolationForest + fast z-score baseline per asset type
serving/inference.py         in-memory inference engine (ONNX Runtime + LightGBM)
serving/app.py                FastAPI: REST + WebSocket, live simulator, work orders
dashboard/app.py              Streamlit fleet dashboard (polls the API)
serving/model_store/          trained model artifacts (generated, gitignored-worthy)
```

### Why this covers the 5 requirements

1. **Continuous monitoring / early failure detection** -- `LiveTelemetrySimulator`
   emulates an onboard sensor gateway streaming vibration/temperature/pressure/
   cycle data; `InferenceEngine` scores every event in real time.
2. **Anomaly detection module** -- two-tier design (see below), covering
   engines (traction motors), brakes and bogies with distinct sensor profiles.
3. **RUL estimation** -- per-asset-type LightGBM regressor trained on
   run-to-failure trajectories, RUL capped at 300 cycles (matches how far out
   a maintenance plan actually needs to look).
4. **Maintenance action dashboard** -- Streamlit fleet view: prioritized
   queue, RUL distributions, health mix, live latency SLA panel.
5. **MRO/EAM integration** -- `POST/GET /workorders` stub + auto work-order
   creation when priority crosses a threshold, shaped to be swapped for a
   real SAP PM / IBM Maximo adapter without changing the scoring path.

## Real-time data

`data/simulate_telemetry.py` models three asset classes (bogie bearings,
brake units, traction motors) with realistic multi-channel sensors and an
exponential "knee curve" degradation trajectory (same shape used in NASA
C-MAPSS turbofan degradation data), so RUL is genuinely learnable from
rolling trend/variance features rather than memorized.

`LiveTelemetrySimulator` streams one event at a time across a simulated
fleet, matching what a real MQTT/Kafka telemetry feed would push, and
randomly injects sudden-fault degradation into a few assets so the anomaly
detector and dashboard have something to catch live.

## Why it's fast enough for production

Predictive maintenance on a live sensor stream has a hard latency budget --
scoring must keep up with the ingestion rate indefinitely, not just work in
a notebook. Three choices make the hot path (`InferenceEngine.process_event`)
fast:

- **ONNX Runtime for RUL.** LightGBM is trained normally, then exported to
  ONNX and served through `onnxruntime` (single-threaded session -- avoids
  thread-pool dispatch overhead on tiny single-row inputs). Falls back to
  the native LightGBM booster automatically if the ONNX toolchain isn't
  available.
- **O(1) incremental features.** `RollingFeatureBuffer` keeps a bounded
  per-asset window with running sums (Welford-style), so feature extraction
  cost never grows with how long an asset has been streaming -- no rescans.
- **Two-tier anomaly detection.** IsolationForest is an excellent
  unsupervised multivariate detector, but scikit-learn's per-tree
  `decision_path` walk has ~20ms of *fixed* overhead per call regardless of
  batch size (measured: 22ms for 1 row vs. 0.1ms/row at a 200-row batch). That
  fixed cost is incompatible with a per-event hot path. So:
  - **Real-time gate** (per event, hot path): an O(n_sensors) z-score
    statistic against each sensor's healthy baseline -- microseconds.
  - **Deep scan** (batched, every 5s, `run_deep_scan()`): IsolationForest
    scores every currently-buffered asset in one batched call per asset
    type, amortizing its fixed overhead across the whole fleet. Feeds a
    `deep_anomaly_score` field the dashboard shows alongside the real-time
    flag -- a slower, higher-fidelity second opinion, the same pattern real
    fleets use (fast rule-based alerting + periodic deep ML sweeps).

Measured end-to-end on a laptop CPU (`GET /metrics/latency` after the demo
fleet has been streaming): **p50 ~0.7ms, p99 ~3ms** per event, including
both the RUL and anomaly path -- comfortably inside a real-time SLA even at
hundreds of events/sec fleet-wide.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# Train both models (regenerates the synthetic dataset each run)
./.venv/Scripts/python -m src.train_rul
./.venv/Scripts/python -m src.train_anomaly
```

This writes `serving/model_store/{rul,anomaly}_<asset_type>.{txt,onnx,joblib}`
plus `rul_metadata.json` / `anomaly_metadata.json`.

## Run

```bash
# Terminal 1: serving API (also runs the live fleet simulator in the background)
./.venv/Scripts/python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000

# Terminal 2: dashboard
./.venv/Scripts/python -m streamlit run dashboard/app.py
```

Open the Streamlit URL it prints. The dashboard polls `http://127.0.0.1:8000`
by default (editable in the sidebar).

### Key endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | push one real telemetry reading, get a prediction back |
| `GET /fleet/status` | latest snapshot for every asset, priority-sorted |
| `GET /metrics/latency` | rolling p50/p95/p99 inference latency |
| `POST /workorders`, `GET /workorders` | MRO/EAM integration stub |
| `WS /ws/stream` | live prediction feed for custom front ends |

## Notes / next steps for a full product

- Swap `LiveTelemetrySimulator` for a real Kafka/MQTT consumer calling the
  same `InferenceEngine.process_event` -- the scoring code doesn't change.
- Swap the `/workorders` stub for a real SAP PM / Maximo adapter.
- Add periodic retraining as real run-to-failure and maintenance-log data
  accumulates, replacing the synthetic dataset.
