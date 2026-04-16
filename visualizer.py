"""
visualizer.py
--------------
Generates charts and visual summaries from session metrics.

Outputs:
  - Attention timeline line chart (PNG)
  - State distribution pie chart (PNG)
  - Engagement heatmap over time (PNG)
  - Combined dashboard figure (PNG)

All charts are saved to the reports/ directory and can be embedded in reports.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Optional

from metrics_engine import SessionMetrics, FrameMetrics
from attention_classifier import AttentionState


# ---------------------------------------------------------------------------
# Color palette (consistent with AttentionState)
# ---------------------------------------------------------------------------

STATE_COLORS_HEX = {
    "Focused":    "#2ecc71",
    "Listening":  "#3498db",
    "Unfocused":  "#f39c12",
    "Distracted": "#e74c3c",
    "Sleepy":     "#9b59b6",
}

BACKGROUND_COLOR = "#1a1a2e"
PANEL_COLOR      = "#16213e"
TEXT_COLOR       = "#e0e0e0"
ACCENT_COLOR     = "#00d4ff"
GRID_COLOR       = "#2a2a4a"


def _apply_dark_theme(fig, axes):
    """Apply a dark theme to all axes in the figure."""
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


# ---------------------------------------------------------------------------
# Individual Charts
# ---------------------------------------------------------------------------

def plot_attention_timeline(metrics: SessionMetrics, save_path: Optional[str] = None):
    """Line chart showing per-minute engagement score over the session."""
    if not metrics.attention_timeline:
        print("[Visualizer] No timeline data to plot.")
        return None

    minutes  = [t["minute"] for t in metrics.attention_timeline]
    scores   = [t["avg_score"] for t in metrics.attention_timeline]
    dom_states = [t["dominant_state"] for t in metrics.attention_timeline]

    fig, ax = plt.subplots(figsize=(12, 4))
    _apply_dark_theme(fig, ax)

    # Filled area under line
    ax.fill_between(minutes, scores, alpha=0.2, color=ACCENT_COLOR)
    ax.plot(minutes, scores, color=ACCENT_COLOR, linewidth=2.5, zorder=5)

    # Color scatter by dominant state
    for m, s, ds in zip(minutes, scores, dom_states):
        color = STATE_COLORS_HEX.get(ds, "#ffffff")
        ax.scatter(m, s, color=color, s=60, zorder=6, edgecolors="white", linewidth=0.5)

    # Reference lines
    ax.axhline(70, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.6, label="High (70)")
    ax.axhline(45, color="#f39c12", linestyle="--", linewidth=1, alpha=0.6, label="Moderate (45)")
    ax.axhline(metrics.class_attention_score, color="white", linestyle=":",
               linewidth=1.5, alpha=0.8, label=f"Session avg ({metrics.class_attention_score:.0f})")

    ax.set_xlim(min(minutes) - 0.5, max(minutes) + 0.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Minute", fontsize=11)
    ax.set_ylabel("Engagement Score", fontsize=11)
    ax.set_title("Attention Timeline — Per-Minute Engagement", fontsize=13, fontweight="bold")

    legend = ax.legend(loc="lower right", facecolor=PANEL_COLOR,
                       edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        print(f"[Visualizer] Timeline chart saved: {save_path}")

    return fig


def plot_state_distribution(metrics: SessionMetrics, save_path: Optional[str] = None):
    """Donut chart of time spent in each attention state."""
    dist = metrics.state_distribution
    labels  = [k for k, v in dist.items() if v > 0]
    values  = [v for v in dist.values() if v > 0]
    colors  = [STATE_COLORS_HEX.get(l, "#888888") for l in labels]

    if not values:
        print("[Visualizer] No state distribution data.")
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    _apply_dark_theme(fig, ax)

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"width": 0.55, "edgecolor": BACKGROUND_COLOR, "linewidth": 2},
        pctdistance=0.75,
    )

    for at in autotexts:
        at.set_color(TEXT_COLOR)
        at.set_fontsize(10)

    legend_patches = [
        mpatches.Patch(color=c, label=f"{l}  ({v:.1f}%)")
        for l, c, v in zip(labels, colors, values)
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=10)

    # Centre label
    ax.text(0, 0, f"CAS\n{metrics.class_attention_score:.0f}",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=ACCENT_COLOR)

    ax.set_title("Attention State Distribution", fontsize=13, fontweight="bold",
                 color=TEXT_COLOR, pad=15)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        print(f"[Visualizer] Distribution chart saved: {save_path}")

    return fig


def plot_engagement_heatmap(metrics: SessionMetrics, save_path: Optional[str] = None):
    """
    Horizontal heatmap of frame-level engagement scores over time.
    Each column = one frame, color = engagement score.
    """
    if not metrics.frame_metrics:
        print("[Visualizer] No frame metrics for heatmap.")
        return None

    scores = np.array([fm.avg_score for fm in metrics.frame_metrics])
    times  = np.array([fm.timestamp_sec for fm in metrics.frame_metrics])

    # Reshape into a 2D array for heatmap (1 row = temporal slice)
    n = len(scores)
    fig, ax = plt.subplots(figsize=(14, 2.5))
    _apply_dark_theme(fig, ax)

    im = ax.imshow(
        scores.reshape(1, n),
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        extent=[times[0], times[-1], 0, 1],
        interpolation="nearest",
    )

    # Mark peak distraction windows
    for w in metrics.peak_distraction_windows:
        ax.axvspan(w["start_sec"], w["end_sec"], alpha=0.35,
                   color="#ff4444", label="Low engagement window")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, fraction=0.02)
    cbar.set_label("Engagement", color=TEXT_COLOR, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_title("Engagement Heatmap", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        print(f"[Visualizer] Heatmap saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Combined Dashboard
# ---------------------------------------------------------------------------

def generate_dashboard(
    metrics: SessionMetrics,
    output_dir: str = "reports",
    session_label: str = None,
) -> str:
    """
    Generate a combined 3-panel dashboard PNG.
    Returns path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)

    label = session_label or metrics.session_id
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.06, right=0.97, top=0.88, bottom=0.1)

    # ---- KPI header -------------------------------------------------------
    fig.text(0.5, 0.94,
             f"Classroom Engagement Dashboard  —  {label}",
             ha="center", fontsize=16, fontweight="bold", color=ACCENT_COLOR)

    kpi_text = (
        f"CAS: {metrics.class_attention_score:.1f}/100   |   "
        f"FSI: {metrics.focus_stability_index:.2f}   |   "
        f"PES: {metrics.participation_equity_score:.2f}   |   "
        f"Disengaged: {metrics.disengagement_duration_pct:.1f}%   |   "
        f"Level: {metrics.engagement_level}"
    )
    fig.text(0.5, 0.90, kpi_text, ha="center", fontsize=11, color=TEXT_COLOR)

    # ---- Panel 1: Timeline (full width top) --------------------------------
    ax_timeline = fig.add_subplot(gs[0, :])
    _apply_dark_theme(fig, ax_timeline)

    if metrics.attention_timeline:
        minutes = [t["minute"] for t in metrics.attention_timeline]
        scores  = [t["avg_score"] for t in metrics.attention_timeline]
        dom     = [t["dominant_state"] for t in metrics.attention_timeline]

        ax_timeline.fill_between(minutes, scores, alpha=0.15, color=ACCENT_COLOR)
        ax_timeline.plot(minutes, scores, color=ACCENT_COLOR, linewidth=2, zorder=5)
        for m, s, ds in zip(minutes, scores, dom):
            ax_timeline.scatter(m, s, color=STATE_COLORS_HEX.get(ds, "#fff"),
                                s=55, zorder=6, edgecolors="white", linewidth=0.4)
        ax_timeline.axhline(70, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.5)
        ax_timeline.axhline(45, color="#f39c12", linestyle="--", linewidth=1, alpha=0.5)
        ax_timeline.axhline(metrics.class_attention_score, color="white",
                            linestyle=":", linewidth=1.5, alpha=0.7)
        ax_timeline.set_ylim(0, 105)
        ax_timeline.set_xlabel("Minute", color=TEXT_COLOR)
        ax_timeline.set_ylabel("Score", color=TEXT_COLOR)
        ax_timeline.set_title("Attention Timeline", color=TEXT_COLOR, fontsize=12)

    # ---- Panel 2: State distribution donut --------------------------------
    ax_pie = fig.add_subplot(gs[1, 0])
    _apply_dark_theme(fig, ax_pie)
    ax_pie.set_aspect("equal")
    ax_pie.axis("off")

    dist   = metrics.state_distribution
    labels = [k for k, v in dist.items() if v > 0]
    values = [v for v in dist.values() if v > 0]
    colors = [STATE_COLORS_HEX.get(l, "#888") for l in labels]

    if values:
        wedges, _, autotexts = ax_pie.pie(
            values, autopct="%1.0f%%", colors=colors, startangle=140,
            wedgeprops={"width": 0.55, "edgecolor": BACKGROUND_COLOR, "linewidth": 1.5},
            pctdistance=0.76,
        )
        for at in autotexts:
            at.set_color(TEXT_COLOR); at.set_fontsize(8)

        ax_pie.text(0, 0, f"{metrics.class_attention_score:.0f}",
                    ha="center", va="center", fontsize=20, fontweight="bold", color=ACCENT_COLOR)

        legend_patches = [mpatches.Patch(color=c, label=l)
                          for l, c in zip(labels, colors)]
        ax_pie.legend(handles=legend_patches, loc="lower center",
                      bbox_to_anchor=(0.5, -0.18), ncol=2,
                      facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
                      labelcolor=TEXT_COLOR, fontsize=8)
        ax_pie.set_title("State Distribution", color=TEXT_COLOR, fontsize=12, pad=10)

    # ---- Panel 3: KPI bar chart -------------------------------------------
    ax_kpi = fig.add_subplot(gs[1, 1])
    _apply_dark_theme(fig, ax_kpi)

    kpi_names  = ["Attention\nScore", "Focus\nStability", "Participation\nEquity",
                   "Engagement\n(inverse diseng.)"]
    kpi_values = [
        metrics.class_attention_score,
        metrics.focus_stability_index * 100,
        metrics.participation_equity_score * 100,
        100 - metrics.disengagement_duration_pct,
    ]
    bar_colors = ["#00d4ff", "#2ecc71", "#f39c12", "#9b59b6"]

    bars = ax_kpi.bar(kpi_names, kpi_values, color=bar_colors, width=0.5,
                      edgecolor=BACKGROUND_COLOR, linewidth=1.2)
    ax_kpi.set_ylim(0, 110)
    ax_kpi.set_ylabel("Score (0–100)", color=TEXT_COLOR, fontsize=9)
    ax_kpi.set_title("Key Performance Indicators", color=TEXT_COLOR, fontsize=12)
    ax_kpi.tick_params(axis="x", colors=TEXT_COLOR, labelsize=8)

    for bar, val in zip(bars, kpi_values):
        ax_kpi.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9,
                    fontweight="bold")

    out_path = os.path.join(output_dir, f"{metrics.session_id}_dashboard.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"[Visualizer] Dashboard saved: {out_path}")
    return out_path
