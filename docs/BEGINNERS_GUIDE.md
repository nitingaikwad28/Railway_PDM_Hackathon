# Beginner's Guide

A from-scratch explanation of this project for someone new to **machine
learning**, **railway/aerospace maintenance**, **Python**, or **servers**.
No prior background assumed. Read top to bottom, or jump to a section.

- [1. The problem in plain words](#1-the-problem-in-plain-words)
- [2. Key ideas & vocabulary](#2-key-ideas--vocabulary)
- [3. How the whole system fits together](#3-how-the-whole-system-fits-together)
- [4. A tour of every file](#4-a-tour-of-every-file)
- [5. The machine-learning part, gently](#5-the-machine-learning-part-gently)
- [6. The "server" part, gently](#6-the-server-part-gently)
- [7. Run it yourself, step by step](#7-run-it-yourself-step-by-step)
- [8. How we measure if it's any good](#8-how-we-measure-if-its-any-good)
- [9. FAQ](#9-faq)

---

## 1. The problem in plain words

A train (or aircraft) is full of parts that slowly wear out: bearings,
brakes, motors. If a part fails while in service, you get an unplanned
breakdown — expensive, disruptive, and sometimes dangerous.

There are two bad ways to handle this:

1. **Run to failure** — wait until it breaks. Cheapest until the day it
   strands a train on the mainline.
2. **Fixed schedule** — replace every part every N months no matter what.
   Safe, but wasteful: you throw away parts that still had plenty of life.

**Predictive maintenance (PdM)** is the smart third way: watch the part's
sensors, notice when it *starts* to degrade, estimate how much life is left,
and fix it *just before* it would fail — not too early, not too late.

That's exactly what this project does, for a simulated fleet of trains.

---

## 2. Key ideas & vocabulary

| Term | Plain meaning |
|---|---|
| **Asset** | A physical part we monitor (a bogie bearing, a brake unit, a traction motor). "Bogie" = the wheel-and-axle assembly under a rail car. "Traction motor" = the electric motor that drives the wheels. |
| **Telemetry / sensor data** | Numbers streaming off the asset: vibration, temperature, pressure, current, etc. Like a fitness tracker for machines. |
| **Channel** | One sensor's stream (e.g. the "temperature" channel). |
| **Operating cycle** | One unit of use (think "one trip" or "one hour of running"). We measure age and remaining life in cycles. |
| **Degradation** | The gradual worsening of a part as it wears. Usually slow at first, then accelerating — a "knee curve". |
| **Anomaly** | Sensor behavior that no longer looks like a healthy part. The first sign of trouble. |
| **RUL (Remaining Useful Life)** | How many more cycles the part can run before it needs maintenance. The headline number a planner wants. |
| **Baseline** | What "healthy" looks like, learned from data. We flag deviations *from* the baseline. |
| **Model** | A mathematical function, fitted to past data, that makes a prediction (here: "is this an anomaly?" and "what's the RUL?"). |
| **Feature** | An input number we feed the model. We don't feed raw sensors directly; we compute richer signals first (see §5). |
| **False alarm** | Crying wolf — flagging a healthy part as degrading. Too many and people stop trusting the system. |
| **Lead time** | How far *ahead* of failure we raise the alarm. More lead time = more time to plan a fix. |

---

## 3. How the whole system fits together

Four moving pieces:

```
 (1) DATA GENERATOR            (2) MODEL TRAINING           (3) SERVING API
 fake but realistic     ──▶    learns "healthy" +     ──▶   loads trained models,
 sensor streams               "how life decays"             scores live data fast
      │                                                          │
      │                                                          ▼
      └───────────── live stream of readings ───────────▶  (4) DASHBOARD
                                                           shows fleet health,
                                                           alerts, RUL, priorities
```

1. **Data generator** invents realistic sensor data (we don't have a real
   train fleet handy, so we simulate one — see §5 for why this is legitimate).
2. **Training** looks at lots of historical "run-to-failure" examples and
   learns two things: what healthy looks like, and how sensor trends map to
   remaining life.
3. **Serving API** is a small always-on program that holds the trained models
   in memory and answers "here's a new reading, what's the health/RUL?" in a
   couple of milliseconds.
4. **Dashboard** is the screen a maintenance planner actually looks at.

---

## 4. A tour of every file

```
Hackathon_Railway_PDM/
├── data/
│   └── simulate_telemetry.py   ← invents realistic sensor data + a live feed
├── src/
│   ├── features.py             ← turns raw sensors into model inputs ("features")
│   ├── train_rul.py            ← trains the Remaining-Useful-Life predictor
│   ├── train_anomaly.py        ← trains the "is this healthy?" detector
│   └── evaluate.py             ← measures how good the models are
├── serving/
│   ├── inference.py            ← the fast scoring engine (models live here)
│   ├── app.py                  ← the web API around the engine
│   └── model_store/            ← saved trained models + benchmark report
├── dashboard/
│   └── app.py                  ← the visual fleet dashboard
├── docs/                       ← you are here
├── requirements.txt            ← the Python libraries this project needs
└── README.md                   ← quick-start + architecture summary
```

You only ever *run* three things: the two training scripts, the serving API,
and the dashboard. Everything else is supporting code they import.

---

## 5. The machine-learning part, gently

### Why simulated data is OK here

Real run-to-failure datasets for rail are proprietary and scarce. So we
generate synthetic data — but not random noise. Each simulated part follows a
physically-motivated **degradation curve**: sensors stay flat and noisy while
healthy, then trend upward (or downward) with an accelerating "knee" as the
part approaches end-of-life. This is the same shape as NASA's well-known
**C-MAPSS** turbofan degradation dataset. Because the degradation is *real*
in the data, the models have to *learn* it rather than memorize noise — and
the same code runs unchanged on real telemetry later (see the integration
roadmap).

### Features: why we don't feed raw sensors

A single temperature reading of 60°C tells you little. But "temperature has
been *rising* for the last 20 cycles and its variability just doubled" is a
strong degradation signal. So for each sensor we compute, over a sliding
window of recent readings:

- **rolling mean** — the recent average (smooths out noise),
- **rolling standard deviation** — how jumpy it's been (wear often adds jitter),
- **rolling slope** — is it trending up or down, and how fast,
- **delta** — the change since the previous reading.

These derived numbers are the **features**. `src/features.py` computes them.
The exact same feature math runs in training and in live serving — otherwise
the model would see different inputs in the two settings ("train/serve skew").

### Two models, two jobs

**Anomaly detection ("is this still healthy?")** — `src/train_anomaly.py`.
We train an **Isolation Forest**, an algorithm that learns the shape of
"normal" from healthy data *only* (it needs no examples of failures, which
real fleets rarely have enough of). Anything that doesn't fit the normal
shape gets a low score → flagged. For the *real-time* path we also compute a
cheaper **z-score** statistic (how many standard deviations each sensor is
from its healthy baseline) so we can flag instantly on every event; the
Isolation Forest runs as a slower, deeper second-opinion sweep. (See the
README's "Why it's fast enough" section for the latency reasoning.)

**RUL estimation ("how much life is left?")** — `src/train_rul.py`. We train
a **LightGBM** model — a "gradient-boosted decision tree" regressor. Think of
it as a large committee of simple yes/no rules ("is the vibration slope above
X? is the temperature variance above Y?") that together output a number: the
predicted remaining cycles. It trains in seconds on a laptop and predicts in
microseconds. We **cap** the target at 300 cycles because predicting "5000
cycles left" precisely is neither possible nor useful — planners only care
once a part enters its final few hundred cycles.

### What "training" actually does

Training = show the model many past examples where we know the right answer
(the true RUL, or known-healthy data), and let it adjust its internal numbers
until its predictions match. We then **save** the fitted model to disk
(`serving/model_store/`). Serving just **loads** that saved model — no
learning happens live.

---

## 6. The "server" part, gently

### What is a server / API?

A **server** here is just a Python program that stays running and waits for
requests. An **API** (Application Programming Interface) is the menu of
requests it understands. Ours is built with **FastAPI**, a popular Python web
framework.

You talk to it over HTTP — the same protocol your browser uses. Examples:

- `GET /health` → "are you alive?" → `{"status": "ok", ...}`
- `POST /ingest` → "here's a sensor reading, score it" → prediction back
- `GET /fleet/status` → "give me every asset's latest health, worst first"
- `GET /metrics/model` → "how accurate are your models?" (the benchmark report)

`GET` = "read something." `POST` = "send something / make something happen."

### Why a server at all — why not just a script?

Because predictions have to happen **continuously and fast** as data streams
in, potentially hundreds of readings per second across the fleet, forever. A
server loads the (relatively slow-to-load) models **once** at startup and
then answers each request in ~1 millisecond. Restarting a script per reading
would be thousands of times slower.

`serving/inference.py` is the brain (loads models, scores a reading).
`serving/app.py` wraps that brain in the web API and also runs a background
loop that feeds it the simulated live fleet, so there's always something to
look at.

### The dashboard

`dashboard/app.py` uses **Streamlit**, a tool that turns a Python script into
a web page with almost no web code. It periodically calls the server's API
(`/fleet/status`, `/metrics/model`, `/workorders`) and draws tables and
charts. It holds *no* intelligence itself — it's a window onto the server.

---

## 7. Run it yourself, step by step

You need **Python 3.10+** installed. Commands below are for Windows
(PowerShell / Git Bash); on macOS/Linux replace `./.venv/Scripts/` with
`./.venv/bin/`.

**Step 1 — install the libraries** (one time):

```bash
python -m venv .venv                       # make an isolated Python sandbox
./.venv/Scripts/pip install -r requirements.txt
```

A "virtual environment" (`.venv`) keeps this project's libraries separate from
the rest of your computer. `requirements.txt` lists what to install.

**Step 2 — train the models** (a couple of minutes):

```bash
./.venv/Scripts/python -m src.train_rul       # trains the RUL predictor
./.venv/Scripts/python -m src.train_anomaly   # trains the anomaly detector
```

This writes trained model files into `serving/model_store/`.

**Step 3 — measure how good they are** (optional but recommended):

```bash
./.venv/Scripts/python -m src.evaluate
```

Prints benchmark tables and saves `evaluation_report.json`.

**Step 4 — start the server** (leave it running):

```bash
./.venv/Scripts/python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

Visit http://127.0.0.1:8000/docs in a browser — FastAPI auto-generates an
interactive page where you can click every endpoint. `uvicorn` is the program
that actually runs the FastAPI app.

**Step 5 — start the dashboard** (in a second terminal):

```bash
./.venv/Scripts/python -m streamlit run dashboard/app.py
```

It opens a browser tab showing the live fleet. Done.

---

## 8. How we measure if it's any good

We never trust a model on the data it trained on — that's like grading
students on questions they've already seen. Instead `src/evaluate.py` builds a
**fresh held-out fleet** (new random seed, assets the models never saw) and
replays each part's whole life through the real serving path. Then it reports:

**For RUL (regression accuracy):**

- **MAE (Mean Absolute Error)** — on average, how many cycles off is the
  prediction? Lower is better. *(We get ~1.7 cycles overall.)*
- **RMSE** — like MAE but punishes big misses more.
- **R²** — fraction of the variation the model explains, 0–1. *(We get ~0.97.)*
- **Error by RUL band** — accuracy specifically when the part is *close* to
  failure, which is what matters most for planning.

**For anomaly detection:**

- **Detection lead time** — how many cycles of early warning we get before
  end-of-life. More = more time to act. *(Median ~1390 cycles.)*
- **False-alarm rate** — fraction of genuinely-healthy readings we wrongly
  flag. Lower = more trustworthy. *(~1%.)*
- **Detection rate** — of all failing parts, how many did we catch in time?
  *(100% in the benchmark.)*
- **Precision / Recall** — of our alarms, how many were real (precision); of
  the real problems, how many we caught (recall).

These are the numbers in [`docs/EVALUATION.md`](EVALUATION.md) and on the
dashboard's "Model quality benchmarks" panel.

---

## 9. FAQ

**Q: Is the data real?**
No — it's simulated, but with realistic physics-based degradation (§5). The
code path is built so that swapping in real sensor streams changes only the
ingestion adapter, not the models (see the integration roadmap).

**Q: What's a "cycle"?**
Our unit of usage/age. In a real deployment it might be an operating hour, a
trip, or a duty cycle — whatever the maintenance schedule is denominated in.

**Q: Why two anomaly methods (z-score and Isolation Forest)?**
Speed vs. depth. The z-score is instant and runs on every event; the
Isolation Forest is a richer multivariate check that's too slow per-event, so
it runs as a periodic batched sweep. Together: fast alarms + deep second
opinion.

**Q: Why cap RUL at 300 cycles?**
Because predicting "3,412 cycles left" precisely is impossible and pointless.
Planners act in the final stretch, so we focus the model's accuracy there.

**Q: Where do I see everything working?**
The Streamlit dashboard (Step 5) — fleet table, alerts, RUL charts, work
orders, and the benchmark panel — all live.

**Q: I'm an ML person — where's the interesting engineering?**
The train/serve feature-parity design in `features.py` (batch rolling ops for
training, O(1) incremental buffers for serving) and the two-tier anomaly
architecture that keeps p99 latency ~3ms. See the README.
