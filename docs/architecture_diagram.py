"""
Render the system architecture diagram as a PNG (used in the combined PDF and
viewable on its own). Pure matplotlib, no live services required.

Run:  python docs/architecture_diagram.py
Out:  docs/architecture_diagram.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "architecture_diagram.png")

# palette
C_DATA = "#2E86AB"
C_TRAIN = "#6A4C93"
C_STORE = "#1B998B"
C_SERVE = "#E1701A"
C_UI = "#C5283D"
C_EXT = "#5C6672"
TEXT = "#1a1a1a"


def box(ax, cx, cy, w, h, title, subtitle, color):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, edgecolor=color, facecolor=color + "1A", clip_on=False))
    ax.text(cx, cy + 0.14, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TEXT)
    ax.text(cx, cy - 0.26, subtitle, ha="center", va="center",
            fontsize=7.8, color="#333333")


def arrow(ax, x1, y1, x2, y2, color="#444444", label=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        linewidth=1.6, color=color, shrinkA=2, shrinkB=2, clip_on=False))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center",
                va="center", fontsize=7.2, color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6, 7.62, "Railway Predictive Maintenance — System Architecture",
            ha="center", fontsize=14, fontweight="bold", color=TEXT)

    # lane labels
    ax.text(0.15, 6.35, "OFFLINE  (train + evaluate)", fontsize=8.5,
            fontweight="bold", color="#888888", rotation=90, va="center")
    ax.text(0.15, 2.55, "ONLINE  (real-time serving)", fontsize=8.5,
            fontweight="bold", color="#888888", rotation=90, va="center")

    W, H = 2.35, 1.05

    # --- OFFLINE lane (top) ---
    box(ax, 2.0, 6.3, W, H, "Data Generator", "simulate_telemetry.py\nrun-to-failure dataset", C_DATA)
    box(ax, 5.0, 6.3, W, H, "Feature Engineering", "features.py\nmean / std / slope / delta", C_TRAIN)
    box(ax, 8.0, 6.3, W, H, "Model Training", "train_rul (LightGBM→ONNX)\ntrain_anomaly (IForest+z)", C_TRAIN)
    box(ax, 11.0, 6.3, W * 0.92, H, "Model Store", "*.onnx  *.txt\n*.joblib  +metadata", C_STORE)

    box(ax, 8.0, 4.35, W, H, "Evaluation", "evaluate.py\nMAE · lead time · false-alarm", C_EXT)

    arrow(ax, 3.18, 6.3, 3.83, 6.3, C_DATA, "train data")
    arrow(ax, 6.18, 6.3, 6.83, 6.3, C_TRAIN)
    arrow(ax, 9.18, 6.3, 9.75, 6.3, C_TRAIN, "save")
    arrow(ax, 10.6, 5.77, 8.5, 4.88, C_STORE, "load", style="-|>")
    arrow(ax, 8.0, 5.77, 8.0, 4.88, C_EXT)

    # --- ONLINE lane (bottom) ---
    box(ax, 2.0, 2.5, W, H, "Live Telemetry", "stream() — MQTT / Kafka\nin production", C_DATA)
    box(ax, 5.0, 2.5, W, H, "FastAPI Service", "serving/app.py\nREST + WebSocket", C_SERVE)
    box(ax, 8.0, 2.5, W, H, "Inference Engine", "inference.py\nONNX + z-score + IForest", C_SERVE)
    box(ax, 11.0, 2.5, W * 0.92, H, "Dashboard", "Streamlit\nhealth · alerts · RUL", C_UI)

    box(ax, 8.0, 0.62, W, H * 0.92, "MRO / EAM", "work orders →\nSAP PM · IBM Maximo", C_EXT)

    arrow(ax, 2.0, 5.77, 2.0, 3.03, C_DATA, None)
    ax.text(2.12, 4.4, "live stream", fontsize=7.2, color=C_DATA, style="italic", rotation=90, va="center")
    arrow(ax, 3.18, 2.5, 3.83, 2.5, C_SERVE, "ingest")
    arrow(ax, 6.18, 2.5, 6.83, 2.5, C_SERVE)
    arrow(ax, 9.18, 2.5, 9.75, 2.5, C_UI, "predictions")
    arrow(ax, 11.0, 5.77, 11.0, 3.03, C_STORE, None)
    ax.text(11.12, 4.4, "load once", fontsize=7.2, color=C_STORE, style="italic", rotation=90, va="center")
    arrow(ax, 8.0, 1.97, 8.0, 1.13, C_EXT, "auto work-order")

    # evaluation report feeds dashboard/API
    arrow(ax, 9.18, 4.35, 11.0, 3.03, C_EXT, "benchmarks", style="-|>")

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
