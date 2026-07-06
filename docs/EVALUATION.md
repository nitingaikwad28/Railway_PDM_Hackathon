# Model Evaluation & Benchmarks

All numbers below are produced by [`src/evaluate.py`](../src/evaluate.py) and
saved to `serving/model_store/evaluation_report.json`. They are also served
live at `GET /metrics/model` and shown on the dashboard's "Model quality
benchmarks" panel.

Regenerate any time with:

```bash
python -m src.evaluate
```

---

## Methodology (why these numbers are trustworthy)

- **Held-out test fleet.** We generate a *fresh* set of run-to-failure assets
  with a different random seed (`2024`) from the one used for training (`42`).
  The models have never seen these assets.
- **Real serving path.** Each asset's entire life is replayed cycle-by-cycle
  through the exact production `InferenceEngine.process_event` — the same code
  that serves live traffic. So the metrics reflect real behaviour, including
  the incremental feature buffers, not an idealized offline pipeline.
- **12 assets per type × 3 asset types = 36 unseen assets**, ~141,600 scored
  cycles total.

---

## 1. RUL accuracy benchmarks

Error is measured in **operating cycles** against the capped RUL target
(cap = 300 cycles — see note below). Lower MAE/RMSE and higher R² are better.

| Asset type | MAE (cycles) | RMSE (cycles) | R² | n (cycles) |
|---|---|---|---|---|
| bogie_bearing | 1.26 | 6.55 | 0.978 | 52,031 |
| brake_unit | 3.45 | 13.37 | 0.950 | 27,514 |
| traction_motor | 1.34 | 7.12 | 0.970 | 62,055 |
| **overall** | **1.72** | **8.53** | **0.967** | **141,600** |

### Accuracy near failure (what actually matters)

A planner cares most about accuracy when a part is *close* to failure. Error
broken down by how much true life remains:

| RUL band (cycles) | bogie MAE | brake MAE | motor MAE |
|---|---|---|---|
| 0–50 (imminent) | 4.56 | 4.17 | 6.86 |
| 50–150 | 11.69 | 17.40 | 13.67 |
| 150–300 | 24.06 | 34.80 | 28.33 |
| 300+ (healthy tail) | 0.12 | 0.37 | 0.20 |

Reading this table:
- In the **final 50 cycles**, predictions are within ~4–7 cycles — tight
  enough to schedule an intervention confidently.
- The **150–300 band** has the largest error: this is the onset of
  degradation, where the future is genuinely most uncertain. Error shrinking
  as failure approaches is exactly the desirable behaviour.
- The **300+ tail** is near-zero because those cycles are all correctly
  predicted at the cap.

> **Why cap RUL at 300 cycles?** Early in a healthy part's life the true RUL
> can be thousands of cycles. Predicting that precisely is impossible and
> operationally useless — nobody schedules maintenance 4,000 cycles out. So
> the model is trained (and scored) to say "≥ 300, plenty of life" until a
> part enters its actionable final window, then to sharpen. This mirrors the
> standard piecewise-linear RUL target used on NASA C-MAPSS.

---

## 2. Anomaly detection metrics (incl. detection lead time & false-alarm rate)

| Asset type | Detection rate | Median lead time (cycles) | Mean lead time | False-alarm rate | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| bogie_bearing | 100% | 1,707 | 1,687 | 1.03% | 0.984 | 0.931 | 0.957 |
| brake_unit | 100% | 819 | 863 | 0.90% | 0.985 | 0.868 | 0.923 |
| traction_motor | 100% | 1,977 | 1,972 | 1.14% | 0.981 | 0.898 | 0.938 |
| **overall** | **100%** | **1,390** | **1,507** | **1.05%** | **0.983** | **0.904** | **0.942** |

### What each metric means

- **Detection rate** — fraction of failing assets flagged (with a sustained
  alarm) *before* end-of-life. **100%** — every failure was caught in the
  benchmark.
- **Detection lead time** — operating cycles of early warning between the
  first sustained alarm and end-of-life. A **median of ~1,390 cycles** means
  we typically raise the flag well before failure, leaving ample planning
  runway. (Lead time varies by asset because each type wears over a different
  lifespan.)
- **False-alarm rate** — fraction of genuinely-healthy readings (first 60% of
  life) that were wrongly flagged. **~1%** — low enough to stay trustworthy;
  the threshold is deliberately set at the 99th percentile of healthy scores.
- **Precision** — of all alarm-cycles, the fraction that occurred while the
  asset was actually degrading (not healthy). **0.98.**
- **Recall** — of all degrading-cycles, the fraction we flagged. **0.90.**

### How "detection" is defined

- **Sustained alarm (debounce):** we require **3 consecutive** flagged cycles
  before declaring a detection, so a single noisy spike doesn't count. This is
  standard practice to suppress jitter.
- **Positive class = "degrading":** for precision/recall, a cycle is labelled
  positive if the asset is past its healthy window (life fraction ≥ 0.6 — the
  same boundary the detector was trained on). Scoring instead against only the
  last 10% of life would unfairly penalize correct *early* warnings as false
  positives.

---

## 3. Inference latency (real-time SLA)

Not part of the offline report, but measured live via `GET /metrics/latency`
while the demo fleet streams:

| Percentile | Latency per event |
|---|---|
| p50 | ~0.6 ms |
| p95 | ~1.8 ms |
| p99 | ~3 ms |

This includes **both** the RUL prediction and the anomaly check for one
reading. See the README's "Why it's fast enough for production" section for
the engineering behind these numbers (ONNX Runtime, O(1) incremental
features, two-tier anomaly detection).

---

## Reproducing / tuning

Key knobs in `src/evaluate.py`:

| Constant | Meaning | Default |
|---|---|---|
| `TEST_SEED` | RNG seed for the held-out fleet | 2024 |
| `N_TEST_ASSETS_PER_TYPE` | assets per type in the test fleet | 12 |
| `IMMINENT_FRACTION` | last-fraction-of-life = "imminent" window | 0.10 |
| `DEBOUNCE` | consecutive flags required to declare detection | 3 |
| `RUL_BANDS` | bands for the near-failure accuracy breakdown | see file |

Raising `DEBOUNCE` reduces false alarms but slightly shortens lead time;
lowering the anomaly threshold (in `src/train_anomaly.py`) does the opposite.
This tradeoff is the classic sensitivity vs. specificity dial a real
deployment would tune to the customer's tolerance for false alarms.
