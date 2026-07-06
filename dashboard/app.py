"""
Fleet-level maintenance dashboard (Streamlit).

Polls the FastAPI serving layer (`serving/app.py`), which is itself being
fed by a live telemetry simulator in the background -- so this dashboard
reflects the same continuously-updating fleet state a real ops/maintenance
planning team would see from onboard sensor feeds.

Run (after the API is up):
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")
REFRESH_SECONDS = st.sidebar.slider("Refresh interval (s)", 1, 10, 2)

st.set_page_config(page_title="Railway PdM Fleet Dashboard", layout="wide")
st.title("Railway Predictive Maintenance -- Fleet Dashboard")
st.caption("Live asset health, anomaly alerts and Remaining-Useful-Life across the fleet")

SEVERITY_COLORS = {"Critical": "#d9363e", "Warning": "#faad14", "Healthy": "#389e0d"}


def severity(row: dict) -> str:
    if row["priority_score"] >= 0.75:
        return "Critical"
    if row["priority_score"] >= 0.45 or row["is_anomaly"]:
        return "Warning"
    return "Healthy"


@st.cache_data(ttl=1)
def fetch_json(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.fragment(run_every=REFRESH_SECONDS)
def live_view() -> None:
    fleet = fetch_json("/fleet/status")
    latency = fetch_json("/metrics/latency")
    workorders = fetch_json("/workorders")

    if isinstance(fleet, dict) and "error" in fleet:
        st.error(f"Cannot reach API at {API_BASE}: {fleet['error']}. "
                  f"Start it with `uvicorn serving.app:app --reload`.")
        return

    df = pd.DataFrame(fleet)
    if df.empty:
        st.info("Waiting for first telemetry events from the live stream...")
        return

    df["severity"] = df.apply(severity, axis=1)

    n_critical = int((df["severity"] == "Critical").sum())
    n_warning = int((df["severity"] == "Warning").sum())
    n_open_wo = sum(1 for w in workorders if w.get("status") == "open") if isinstance(workorders, list) else 0

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Assets tracked", len(df))
    kpi_cols[1].metric("Critical", n_critical)
    kpi_cols[2].metric("Warning", n_warning)
    kpi_cols[3].metric("Open work orders", n_open_wo)
    kpi_cols[4].metric("Inference p50 (ms)", latency.get("p50_ms", "-"))
    kpi_cols[5].metric("Inference p99 (ms)", latency.get("p99_ms", "-"))

    st.subheader("Prioritized maintenance queue")
    show_cols = ["asset_id", "asset_type", "train_id", "cycle", "rul_pred_cycles",
                 "anomaly_score", "is_anomaly", "worst_sensor", "priority_score", "severity"]
    styled = df[show_cols].sort_values("priority_score", ascending=False)

    def _row_style(row):
        color = SEVERITY_COLORS[row["severity"]]
        return [f"background-color: {color}22"] * len(row)

    st.dataframe(styled.style.apply(_row_style, axis=1), width='stretch', height=380)

    left, right = st.columns(2)
    with left:
        st.subheader("RUL by asset type")
        fig = px.box(df, x="asset_type", y="rul_pred_cycles", points="all", color="asset_type")
        st.plotly_chart(fig, width='stretch')
    with right:
        st.subheader("Fleet health mix")
        counts = df["severity"].value_counts().reindex(["Healthy", "Warning", "Critical"]).fillna(0)
        fig2 = px.pie(names=counts.index, values=counts.values,
                       color=counts.index, color_discrete_map=SEVERITY_COLORS, hole=0.45)
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Work orders (MRO / EAM integration stub)")
    if isinstance(workorders, list) and workorders:
        st.dataframe(pd.DataFrame(workorders), width='stretch', height=220)
    else:
        st.caption("No work orders raised yet.")

    with st.expander("Raise a manual work order"):
        asset_choice = st.selectbox("Asset", df["asset_id"].tolist())
        if st.button("Raise work order"):
            try:
                requests.post(f"{API_BASE}/workorders",
                              params={"asset_id": asset_choice, "reason": "manual"}, timeout=3)
                st.success(f"Work order raised for {asset_choice}")
            except Exception as e:
                st.error(str(e))


def model_quality_view() -> None:
    """Static panel showing offline benchmark results (does not need refresh)."""
    report = fetch_json("/metrics/model")
    st.header("Model quality benchmarks")
    if not isinstance(report, dict) or not report.get("available"):
        st.caption("No benchmark report yet. Run `python -m src.evaluate` to generate one.")
        return

    st.caption("Measured on a held-out fleet of unseen run-to-failure assets "
               "(different random seed from training).")

    rul = report["rul_benchmarks"]["overall"]
    anom = report["anomaly_metrics"]["overall"]
    c = st.columns(6)
    c[0].metric("RUL MAE (cycles)", rul["mae_cycles"])
    c[1].metric("RUL R²", rul["r2"])
    c[2].metric("Detection rate", f"{anom['detection_rate']*100:.0f}%")
    c[3].metric("Median lead time", f"{anom['median_lead_time_cycles']:.0f} cyc")
    c[4].metric("False-alarm rate", f"{anom['false_alarm_rate']*100:.2f}%")
    c[5].metric("Precision / Recall", f"{anom['precision']:.2f} / {anom['recall']:.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("RUL accuracy by asset type")
        rul_rows = [{"asset_type": k, **{kk: vv for kk, vv in v.items() if kk != "by_rul_band"}}
                    for k, v in report["rul_benchmarks"].items()]
        st.dataframe(pd.DataFrame(rul_rows), width='stretch', hide_index=True)
    with right:
        st.subheader("Anomaly detection by asset type")
        anom_rows = [{"asset_type": k, **v} for k, v in report["anomaly_metrics"].items()]
        st.dataframe(pd.DataFrame(anom_rows), width='stretch', hide_index=True)


live_view()
st.divider()
model_quality_view()
