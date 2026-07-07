# Railway Predictive Maintenance -- Hackathon MVP

AI-powered predictive maintenance for rail assets: multi-channel anomaly
detection + Remaining-Useful-Life (RUL) estimation, served behind a
low-latency real-time API, with a fleet-level maintenance dashboard, model
benchmarks, and an MRO/EAM integration roadmap. Built against the brief in
[`Railway Hackathon Idea.txt`](Railway%20Hackathon%20Idea.txt).

**New here?** Start with the [Beginner's Guide](docs/BEGINNERS_GUIDE.md) — it
assumes no background in ML, rail/aerospace, Python, or servers.

## Deliverables — where each is implemented

| # | Required deliverable | Where | Evidence |
|---|---|---|---|
| 1 | Working **anomaly detection** prototype on a representative sensor dataset | `src/train_anomaly.py`, `serving/inference.py`; data from `data/simulate_telemetry.py` | Live on dashboard + `POST /ingest`; two-tier z-score + Isolation Forest |
| 2 | **RUL estimation** model with **accuracy benchmarks** | `src/train_rul.py`; benchmarks in `src/evaluate.py` | **MAE 1.72 cycles, R² 0.97** — [EVALUATION.md](docs/EVALUATION.md) |
| 3 | **Maintenance action dashboard** — fleet health + prioritized alerts | `dashboard/app.py` | Prioritized queue, RUL/health charts, work orders, benchmark panel |
| 4 | Model eval metrics incl. **detection lead time** & **false-alarm rate** | `src/evaluate.py`, `GET /metrics/model` | **Lead time median 1,390 cyc, false-alarm 1.05%, detection 100%** — [EVALUATION.md](docs/EVALUATION.md) |
| 5 | **Integration roadmap** for MRO / asset-management connectivity | `docs/INTEGRATION_ROADMAP.md` + `POST/GET /workorders` stub | Phased SAP PM / IBM Maximo plan; working work-order workflow |

## Architecture

```
data/simulate_telemetry.py   synthetic multi-channel telemetry (train + live stream)
src/features.py              feature engineering shared by training & serving
src/train_rul.py             LightGBM RUL regressor per asset type -> ONNX export
src/train_anomaly.py         IsolationForest + fast z-score baseline per asset type
src/evaluate.py              held-out benchmarks: RUL accuracy, lead time, false-alarm rate
serving/inference.py         in-memory inference engine (ONNX Runtime + LightGBM)
serving/app.py                FastAPI: REST + WebSocket, live simulator, work orders, metrics
dashboard/app.py              Streamlit fleet dashboard + model-quality panel (polls the API)
serving/model_store/          trained model artifacts + evaluation_report.json (generated)
docs/                         BEGINNERS_GUIDE, EVALUATION, INTEGRATION_ROADMAP
```

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

# Materialize the telemetry dataset to data/ (parquet + readable CSV samples).
# Optional -- training auto-generates it on first run -- but this is the
# explicit way to produce the representative sensor dataset on disk.
./.venv/Scripts/python -m data.generate_dataset

# Train both models (loads data/training_dataset.parquet, or generates it)
./.venv/Scripts/python -m src.train_rul
./.venv/Scripts/python -m src.train_anomaly

# (optional) reproduce the accuracy / lead-time / false-alarm benchmarks
./.venv/Scripts/python -m src.evaluate
```

This writes `serving/model_store/{rul,anomaly}_<asset_type>.{txt,onnx,joblib}`,
the `*_metadata.json` files, and `evaluation_report.json`.

### Where the data lives

| File | What it is |
|---|---|
| `data/training_dataset.parquet` | Full run-to-failure training set (120 assets, ~455k rows) — what the models train on |
| `data/training_dataset_sample.csv` | Readable sample: one full asset life per type (open in Excel to watch sensors drift as RUL → 0) |
| `data/live_telemetry_sample.csv` | A captured window of the live streaming feed (what `/ingest` receives per event) |

Training loads the parquet if present (fast, <1s) and only regenerates it when
missing, so the saved dataset is exactly what the models are trained on.

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
| `GET /metrics/model` | offline benchmark report (RUL accuracy, lead time, false-alarm rate) |
| `POST /workorders`, `GET /workorders` | MRO/EAM integration stub |
| `WS /ws/stream` | live prediction feed for custom front ends |

Interactive API docs are auto-generated at `http://127.0.0.1:8000/docs`.

## Headline results

| Metric | Value | Details |
|---|---|---|
| RUL MAE (overall) | **1.72 cycles** | R² 0.97; ~4–7 cyc in the final 50 cycles |
| Anomaly detection rate | **100%** | of failing assets caught before end-of-life |
| Median detection lead time | **1,390 cycles** | early warning before failure |
| False-alarm rate | **1.05%** | of healthy readings wrongly flagged |
| Inference latency | **p50 ~0.6ms / p99 ~3ms** | per event, RUL + anomaly combined |

Full methodology and per-asset-type breakdown: [docs/EVALUATION.md](docs/EVALUATION.md).

## Documentation

- [docs/BEGINNERS_GUIDE.md](docs/BEGINNERS_GUIDE.md) — concepts + how to run, for
  newcomers to ML / rail / Python / servers.
- [docs/EVALUATION.md](docs/EVALUATION.md) — benchmark methodology & full results.
- [docs/INTEGRATION_ROADMAP.md](docs/INTEGRATION_ROADMAP.md) — phased plan to
  connect to SAP PM / IBM Maximo and real telemetry.
