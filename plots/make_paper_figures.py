#!/usr/bin/env python
"""
Paper-quality figures for VLC Fusion (ECML).

Regenerates:
  - ATR per-class radar plots (seen / unseen)   [data unchanged: ATR model is the same]
  - Waymo per-class radar plots (seen / unseen)  [UPDATED to best checkpoint epoch_30, GPT-4o, 10 cond]
  - Figure 8b : condition-source swap grouped bars (GPT-4o / Moondream2 / SmolVLM)  [UPDATED to epoch_30]

Outputs PDF (vector, for the paper) + PNG (preview) into this directory.
Run:  /data/ataparia/conda/envs/gen_ai/bin/python make_paper_figures.py
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,          # editable/TrueType fonts in the PDF (no type-3)
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "hatch.linewidth": 0.6,      # thin hatch lines (toned-down)
})

# Colourblind-safe, paper-consistent (ours = magenta/pink, baseline = blue)
VLC   = "#D81B60"   # ours
BASE  = "#1565C0"   # Fusion SSD
GRID  = "#B8BEC7"

def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", stem + ".pdf / .png")


# ============================================================================ RADAR
def radar(stem, axis_labels, vals_vlc, vals_base, rmax, rstep,
          title=None, legend_loc="upper center", legend_anchor=(0.5, -0.06),
          legend_ncol=2, label_fs=20, scale100=False):
    """Generic paper-quality radar. vals already in %, unless scale100."""
    if scale100:
        vals_vlc  = [v * 100 for v in vals_vlc]
        vals_base = [v * 100 for v in vals_base]

    N = len(axis_labels)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]
    a = vals_vlc + vals_vlc[:1]
    b = vals_base + vals_base[:1]

    fig = plt.figure(figsize=(8.2, 8.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)        # first axis at top
    ax.set_theta_direction(-1)            # clockwise

    # ---- grid styling
    ax.set_facecolor("white")
    ax.spines["polar"].set_color(GRID)
    ax.spines["polar"].set_linewidth(1.2)
    ax.grid(color=GRID, linewidth=0.9, alpha=0.7)
    ax.set_axisbelow(True)

    # ---- fills first (under everything), then lines on top.
    # Draw the baseline (blue) line LAST so it stays visible even when our
    # polygon fully encloses it (e.g. Waymo, where VLC > baseline on every axis).
    ax.fill(ang, a, color=VLC,  alpha=0.18, zorder=2)
    ax.fill(ang, b, color=BASE, alpha=0.10, zorder=2)

    ax.plot(ang, a, linewidth=2.8, label="VLC Fusion (ours)", color=VLC,
            marker="o", markersize=8, markeredgecolor="white",
            markeredgewidth=0.9, zorder=4)
    ax.plot(ang, b, linewidth=2.6, label="Fusion SSD", color=BASE,
            marker="s", markersize=7, markeredgecolor="white",
            markeredgewidth=0.9, zorder=6)

    # ---- angular (category) labels
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axis_labels, fontsize=label_fs, fontweight="bold")
    ax.tick_params(axis="x", pad=14)

    # ---- radial ticks
    ax.set_ylim(0, rmax)
    yt = list(range(rstep, rmax + 1, rstep))
    ax.set_yticks(yt)
    ax.set_yticklabels([str(v) for v in yt], fontsize=12, color="#555")
    ax.set_rlabel_position(180 / N)       # tuck radial labels between two spokes

    if title:
        ax.set_title(title, fontsize=22, fontweight="bold", pad=28)

    ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, fontsize=18,
              framealpha=0.95, ncol=legend_ncol, columnspacing=1.3,
              handlelength=1.6, borderpad=0.6, edgecolor=GRID)
    _save(fig, stem)


# ----------------------------------------------------------------- ATR data
# VLC-Fusion line UPDATED to the 2-block / 7-condition model (run_eval_ensemble.py,
# ablation_logs/atr_eval/2b_seen.log + 2b_unseen.log). Fusion SSD baseline is not
# condition-dependent, so it is unchanged.
atr_classes = ["PICKUP", "SUV", "BTR70", "BRDM2", "BMP2", "T72", "ZSU23", "2S3", "MTLB", "D20"]
atr_ssd_seen   = [0.4396, 0.5505, 0.6514, 0.6034, 0.7331, 0.7002, 0.6839, 0.7047, 0.5851, 0.1299]
atr_vlc_seen   = [0.5207, 0.5770, 0.6478, 0.5923, 0.6939, 0.7300, 0.6636, 0.6753, 0.5507, 0.5155]
atr_ssd_unseen = [0.0064, 0.0732, 0.1472, 0.0695, 0.1688, 0.1278, 0.0665, 0.2157, 0.0451, 0.0219]
atr_vlc_unseen = [0.0107, 0.1023, 0.1882, 0.0925, 0.2430, 0.2363, 0.1134, 0.2626, 0.0683, 0.0209]

radar("atr_seen_radar",   atr_classes, atr_vlc_seen,   atr_ssd_seen,   rmax=80, rstep=20,
      legend_anchor=(0.5, -0.05), scale100=True)
radar("atr_unseen_radar", atr_classes, atr_vlc_unseen, atr_ssd_unseen, rmax=30, rstep=10,
      legend_anchor=(0.5, -0.05), scale100=True)

# ------------------------------------------------- Waymo data (UPDATED: epoch_30, GPT-4o, 10 cond)
wy_labels      = ["Vehicle-L1", "Vehicle-L2", "Cyclist-L1", "Cyclist-L2", "Ped.-L1", "Ped.-L2"]
wy_ssd_seen    = [19.7, 16.9, 21.5, 19.7, 37.2, 32.4]      # Fusion SSD baseline (unchanged)
wy_vlc_seen    = [33.96, 29.42, 31.54, 29.02, 45.54, 40.36]  # NEW epoch_30
wy_ssd_unseen  = [21.5, 18.8, 31.1, 29.4, 40.4, 37.4]
wy_vlc_unseen  = [34.13, 30.34, 41.70, 39.51, 48.54, 45.25]  # NEW epoch_30

radar("waymo_seen_radar",   wy_labels, wy_vlc_seen,   wy_ssd_seen,   rmax=50, rstep=10, label_fs=17)
radar("waymo_unseen_radar", wy_labels, wy_vlc_unseen, wy_ssd_unseen, rmax=50, rstep=10, label_fs=17)


# ============================================================================ FIGURE 8b
def figure_8b(stem, metric_name, gpt, moon, smol):
    # Mirrors plotting_reference.ipynb (the ORIGINAL Fig 8b): thin grouped bars,
    # colours blue / forest-green / purple, GPT-4o hatched + opaque, small VLMs alpha 0.7,
    # legend lower-centre. Numbers updated to the epoch_30 checkpoint.
    # (Y-label kept as "3D mAP" per your earlier "remove Score" instruction; reference said "Score".)
    map_scores = {
        "GPT-4o (baseline)": {"Day-Night": gpt[0],  "Dawn/dusk": gpt[1]},
        "Moondream2":        {"Day-Night": moon[0], "Dawn/dusk": moon[1]},
        "SmolVLM":           {"Day-Night": smol[0], "Dawn/dusk": smol[1]},
    }
    models     = list(map_scores.keys())
    variations = list(next(iter(map_scores.values())).keys())
    width, inner_gap, outer_gap = 0.1, 0.01, 0.3
    colors = ["#4C72B0", "forestgreen", "purple"]
    n_models   = len(models)
    group_width = n_models * width + (n_models - 1) * inner_gap
    centers    = np.arange(len(variations)) * (group_width + outer_gap)

    fig, ax = plt.subplots(figsize=(8, 5))
    all_bars = []
    for i, model in enumerate(models):
        heights   = [map_scores[model][v] for v in variations]
        offset    = i * (width + inner_gap) - (group_width - width) / 2
        bars = ax.bar(centers + offset, heights, width=width, color=colors[i], label=model,
                      alpha=0.7 if i != 0 else 1.0, hatch="/" if i == 0 else None)
        all_bars.append(bars)
    for bars in all_bars:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=14)

    ax.set_xticks(centers)
    ax.set_xticklabels(variations)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_ylabel(f"3D {metric_name}", fontsize=20)
    ax.set_xlabel("Scenarios", fontsize=20)
    ax.set_ylim(0, max(gpt + moon + smol) * 1.15)   # headroom for the new (higher) numbers
    ax.legend(fontsize=16, loc="lower center")
    plt.tight_layout()
    _save(fig, stem)

# L1 mAP (primary metric, consistent with Table 2 mAP column) — epoch_30 condition swap
figure_8b("figure8b_condition_swap_mAP",  "mAP",
          gpt=[37.01, 41.46], moon=[36.21, 40.39], smol=[32.13, 36.71])
# L1 mAPH variant (matches the metric the old Fig 8b appears to have used)
figure_8b("figure8b_condition_swap_mAPH", "mAPH",
          gpt=[34.36, 38.34], moon=[33.63, 37.39], smol=[29.64, 33.90])

print("ALL FIGURES DONE ->", OUT)
