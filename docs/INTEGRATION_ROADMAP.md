# MRO / Asset-Management Integration Roadmap

**Goal:** connect this predictive-maintenance engine to the systems a railway
already runs — the MRO (Maintenance, Repair & Overhaul) and EAM (Enterprise
Asset Management) platforms — so that a predicted failure automatically turns
into a scheduled, parts-planned, costed work order, with no analyst
copy-pasting between screens.

New to these acronyms? See the glossary at the bottom.

---

## Where we are today (MVP)

The system already produces the *content* an MRO system needs and exposes it
over a clean HTTP API:

| Capability | Endpoint | Status |
|---|---|---|
| Per-asset health, RUL, anomaly flag | `GET /fleet/status`, `GET /assets/{id}` | ✅ built |
| Prioritized ranking (what to fix first) | `GET /fleet/status` (sorted) | ✅ built |
| Work-order creation | `POST /workorders` | ✅ stub (in-memory) |
| Work-order listing | `GET /workorders` | ✅ stub (in-memory) |
| Auto work-order on urgent asset | internal `_maybe_auto_workorder` | ✅ built |
| Live push feed | `WS /ws/stream` | ✅ built |

"Stub" means the work order is created and tracked in memory to prove the
workflow end-to-end, but it is **not yet written into a real MRO system of
record**. That connection is what this roadmap covers.

The important design property: the *prediction path never changes* across
these phases. Integration is always an **adapter** bolted onto the existing
API — we are adding output destinations, not re-architecting the models.

---

## Phase 1 — Read-only integration (weeks 1–3)

**Objective:** MRO/EAM systems and dashboards can *read* our predictions. No
writes into external systems yet — lowest risk, immediate value.

- **Stable REST contract + OpenAPI spec.** FastAPI already auto-publishes an
  OpenAPI schema at `/openapi.json` and interactive docs at `/docs`. Version
  it (`/v1/...`) and freeze the response shapes.
- **Asset-ID mapping table.** Our `asset_id` (e.g. `train_03_bogie_bearing_0`)
  must map to the customer's canonical equipment number (SAP *Equipment*,
  Maximo *Asset* record). Add a small mapping service / DB table:
  `asset_id ↔ external_equipment_id ↔ functional_location`.
- **Authentication.** Put the API behind an API gateway with OAuth2 client
  credentials or mutual-TLS; add per-consumer rate limits.
- **Historian read (optional).** If the customer stores sensor data in a
  historian (OSIsoft PI, Aveva, InfluxDB), add a reader so we ingest *their*
  telemetry instead of the simulator (see Phase 3).

**Deliverable:** the customer's BI tools and our dashboard both read live
health/RUL from a documented, authenticated `/v1/fleet/status`.

---

## Phase 2 — Work-order write-back (weeks 4–8)

**Objective:** an urgent prediction automatically becomes a real work
notification/order in the customer's MRO system.

- **Replace the in-memory work-order store with an MRO adapter interface:**

  ```
  class WorkOrderSink(Protocol):
      def create(self, wo: WorkOrder) -> str: ...     # returns external WO id
      def update_status(self, ext_id: str, status: str) -> None: ...
  ```

  Ship concrete implementations:
  - **SAP PM / SAP S/4HANA Asset Management** — create a *Notification*
    (type M2) and/or *Maintenance Order* via SAP's OData / BAPI
    (`BAPI_ALM_NOTIF_CREATE`, `BAPI_ALM_ORDER_MAINTAIN`).
  - **IBM Maximo** — create a *Work Order* through the Maximo REST/OSLC API
    (`/maximo/api/os/mxwo`).
  - **Generic webhook** — POST our work-order JSON to any endpoint, for
    systems without a native connector.

- **De-duplication & lifecycle.** Only one open work order per asset per
  root cause (already enforced in the stub via `already_open`). Subscribe to
  status callbacks so closing the WO in SAP/Maximo flows back to us.
- **Human-in-the-loop gate.** Configurable: auto-create vs. "propose and wait
  for planner approval." Railways will start with propose-only.
- **Enrichment.** Attach to each work order: predicted RUL, anomaly evidence
  (which sensor channel deviated — we already compute `worst_sensor`),
  recommended lead time, and a link back to the asset's dashboard view.

**Deliverable:** crossing the urgent-priority threshold creates a real,
de-duplicated, evidence-rich notification in SAP PM or Maximo.

---

## Phase 3 — Ingestion from real telemetry (weeks 6–10, parallel)

**Objective:** stop using the simulator; consume the customer's real sensor
streams.

- **Streaming ingestion.** Swap `LiveTelemetrySimulator` for a Kafka / MQTT /
  AMQP consumer. Each incoming message is normalized to the same `reading`
  dict and passed to the unchanged `InferenceEngine.process_event`.
- **Schema mapping & units.** Map the customer's channel names/units to our
  canonical sensor names per asset type; validate ranges; handle missing
  channels gracefully.
- **Backfill / historian replay.** Re-train on the customer's historical
  run-to-failure data as it becomes available, replacing the synthetic set.
- **Edge option.** For trains with poor connectivity, the same lightweight
  ONNX model can run on an onboard edge gateway and only push
  predictions/alarms upstream.

**Deliverable:** predictions are driven by real onboard sensors, models
retrained on real degradation history.

---

## Phase 4 — Closed-loop planning & scheduling (weeks 10–16)

**Objective:** predictions influence *when and how* maintenance is scheduled,
not just *that* it's needed.

- **Spare-parts / inventory check.** Before proposing a date, query the MRO
  inventory module for part availability; flag long-lead parts early.
- **Scheduling optimization.** Feed RUL windows into the depot scheduler so
  interventions cluster efficiently (group assets on the same train / same
  depot visit) while staying inside each asset's safe RUL window.
- **Feedback loop for model improvement.** When a work order closes, capture
  the actual findings (was the part really degraded? was it a false alarm?).
  This labelled outcome data is fed back to periodically retrain and to track
  live precision / false-alarm rate against the offline benchmarks.
- **Regulatory & audit trail.** Log every prediction→notification→work-order
  chain for safety audits (important in rail).

**Deliverable:** a measurable reduction in unplanned downtime, with a
data-driven feedback loop continuously improving the models.

---

## Integration architecture (target state)

```
 Onboard sensors ─▶ Kafka/MQTT ─▶  Ingestion adapter
                                        │
                                        ▼
                              InferenceEngine.process_event   (UNCHANGED core)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             /fleet/status        WorkOrderSink         /ws/stream
             (dashboards,        ┌────┴────┐          (live consumers)
              BI, EAM read)      ▼         ▼
                              SAP PM     Maximo
                            (notif/WO)  (work order)
```

Everything left of `InferenceEngine` is *ingestion adapters*; everything
right of it is *output adapters*. The models and scoring never change — which
is exactly why integration is low-risk.

---

## Standards & protocols we align to

- **REST + OpenAPI** for synchronous reads/writes (already emitted by FastAPI).
- **OData / BAPI** for SAP PM; **REST/OSLC** for IBM Maximo.
- **Kafka / MQTT / AMQP** for telemetry ingestion.
- **ISO 14224** (reliability & maintenance data for equipment) and
  **MIMOSA OSA-EAI / OSA-CBM** as the vocabulary for condition-based
  maintenance interchange — worth adopting for asset/condition data models so
  we speak the same language as enterprise EAM tools.

---

## Glossary (for newcomers)

- **MRO** — *Maintenance, Repair & Overhaul.* The systems and processes that
  keep physical assets serviceable.
- **EAM** — *Enterprise Asset Management.* Software (SAP PM, IBM Maximo) that
  tracks assets, work orders, spare parts, and maintenance history.
- **Work order / Notification** — the formal record telling a technician to
  inspect or repair a specific asset. A "notification" (SAP) often precedes a
  full "work order."
- **Functional location / Equipment number** — how enterprise systems
  uniquely identify a physical asset and its place in the fleet hierarchy.
- **Historian** — a time-series database purpose-built for industrial sensor
  data (e.g. OSIsoft PI).
- **Edge gateway** — a small computer on the vehicle that can run the model
  locally when the network is unreliable.
