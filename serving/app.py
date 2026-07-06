"""
FastAPI real-time predictive-maintenance serving layer.

Endpoints
---------
GET  /health                  liveness probe
POST /ingest                  push one real telemetry reading -> prediction
GET  /fleet/status            latest health/RUL/anomaly snapshot, fleet-wide,
                               sorted by maintenance priority (for the dashboard)
GET  /assets/{asset_id}       latest snapshot for one asset
GET  /metrics/latency         rolling inference-latency percentiles (p50/p95/p99)
GET  /metrics/model           offline benchmark report (RUL accuracy, lead time,
                               false-alarm rate) from src/evaluate.py
POST /workorders              MRO/EAM integration stub: raise a work order
GET  /workorders              list work orders raised so far
WS   /ws/stream               push live predictions to connected dashboards

A background asyncio task drives `data.simulate_telemetry.LiveTelemetrySimulator`
to emulate a live onboard-sensor feed across a demo fleet, running every event
through the exact same `InferenceEngine.process_event` path that a real
ingestion webhook (`/ingest`) would use -- so the demo and the "production"
code path are identical.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
import uuid
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from data.simulate_telemetry import LiveTelemetrySimulator
from serving.inference import InferenceEngine, PredictionResult

app = FastAPI(title="Railway Predictive Maintenance API", version="0.1.0")

engine = InferenceEngine()
simulator = LiveTelemetrySimulator(n_trains=6, assets_per_train_per_type=2)

latest_status: dict[str, dict] = {}
work_orders: list[dict] = []
latency_samples: deque[float] = deque(maxlen=2000)
ws_clients: set[WebSocket] = set()

SENSOR_FIELDS = {"vibration_rms", "vibration_kurtosis", "temperature_c", "axle_load_kn",
                  "pad_wear_mm", "disc_temperature_c", "brake_pressure_bar", "brake_cycles",
                  "winding_temperature_c", "current_draw_a", "rpm"}


class TelemetryEvent(BaseModel):
    asset_id: str
    asset_type: str
    train_id: str = "unknown"
    cycle: int
    timestamp: float | None = None

    class Config:
        extra = "allow"  # sensor fields vary per asset_type


def _record_result(reading: dict, result: PredictionResult) -> None:
    latest_status[result.asset_id] = {
        **dataclasses.asdict(result),
        "last_reading": reading,
        "deep_anomaly_score": engine.deep_scores.get(result.asset_id),
    }
    latency_samples.append(result.inference_latency_ms)


def _maybe_auto_workorder(result: PredictionResult) -> None:
    """Auto-raise a work order stub when an asset crosses into the urgent band,
    mirroring how this would push into a real MRO/EAM (SAP PM, Maximo, etc.)."""
    if result.priority_score < 0.75:
        return
    already_open = any(
        wo["asset_id"] == result.asset_id and wo["status"] == "open" for wo in work_orders
    )
    if already_open:
        return
    work_orders.append({
        "work_order_id": str(uuid.uuid4())[:8],
        "asset_id": result.asset_id,
        "asset_type": result.asset_type,
        "train_id": result.train_id,
        "reason": "anomaly" if result.is_anomaly else "low_rul",
        "rul_pred_cycles": result.rul_pred_cycles,
        "priority_score": result.priority_score,
        "status": "open",
        "created_at": time.time(),
    })


@app.on_event("startup")
async def start_live_stream() -> None:
    asyncio.create_task(_live_stream_loop())
    asyncio.create_task(_deep_scan_loop())


async def _live_stream_loop() -> None:
    stream = simulator.stream()
    for reading in stream:
        result = engine.process_event(reading)
        _record_result(reading, result)
        _maybe_auto_workorder(result)
        await _broadcast(result)
        await asyncio.sleep(0.02)  # ~50 events/sec fleet-wide, non-blocking


async def _deep_scan_loop() -> None:
    """Periodic batched IsolationForest sweep (see InferenceEngine.run_deep_scan):
    off the per-event hot path, runs every few seconds across whatever assets
    have reported in, and just annotates `deep_anomaly_score` for the dashboard."""
    while True:
        await asyncio.sleep(5.0)
        updated = engine.run_deep_scan()
        for asset_id, score in updated.items():
            if asset_id in latest_status:
                latest_status[asset_id]["deep_anomaly_score"] = score


async def _broadcast(result: PredictionResult) -> None:
    if not ws_clients:
        return
    payload = dataclasses.asdict(result)
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.discard(ws)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "assets_tracked": len(latest_status), "connected_ws": len(ws_clients)}


@app.post("/ingest")
def ingest(event: TelemetryEvent) -> dict:
    reading = event.model_dump()
    reading["timestamp"] = reading.get("timestamp") or time.time()
    result = engine.process_event(reading)
    _record_result(reading, result)
    _maybe_auto_workorder(result)
    return dataclasses.asdict(result)


@app.get("/fleet/status")
def fleet_status() -> list[dict]:
    rows = list(latest_status.values())
    rows.sort(key=lambda r: r["priority_score"], reverse=True)
    return rows


@app.get("/assets/{asset_id}")
def asset_status(asset_id: str) -> dict:
    return latest_status.get(asset_id, {})


@app.get("/metrics/latency")
def latency_metrics() -> dict:
    if not latency_samples:
        return {"count": 0}
    samples = sorted(latency_samples)
    n = len(samples)

    def pct(p: float) -> float:
        return round(samples[min(int(p * n), n - 1)], 3)

    return {
        "count": n,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": round(samples[-1], 3),
    }


@app.get("/metrics/model")
def model_metrics() -> dict:
    """Serve the offline benchmark report produced by `python -m src.evaluate`
    (RUL accuracy, detection lead time, false-alarm rate). Returns a hint if
    the report hasn't been generated yet."""
    report_path = os.path.join(os.path.dirname(__file__), "model_store", "evaluation_report.json")
    if not os.path.exists(report_path):
        return {"available": False, "hint": "Run `python -m src.evaluate` to generate benchmarks."}
    with open(report_path) as f:
        return {"available": True, **json.load(f)}


@app.post("/workorders")
def create_workorder(asset_id: str, reason: str = "manual") -> dict:
    status = latest_status.get(asset_id, {})
    wo = {
        "work_order_id": str(uuid.uuid4())[:8],
        "asset_id": asset_id,
        "asset_type": status.get("asset_type", "unknown"),
        "train_id": status.get("train_id", "unknown"),
        "reason": reason,
        "rul_pred_cycles": status.get("rul_pred_cycles"),
        "priority_score": status.get("priority_score"),
        "status": "open",
        "created_at": time.time(),
    }
    work_orders.append(wo)
    return wo


@app.get("/workorders")
def list_workorders() -> list[dict]:
    return sorted(work_orders, key=lambda w: w["created_at"], reverse=True)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive / ignore client pings
    except WebSocketDisconnect:
        ws_clients.discard(websocket)
