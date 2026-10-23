#!/usr/bin/env python
"""
Figure 9 — mAP vs Number of Conditions (ATR + Waymo).
Rebuilt from plotting_reference.ipynb with two updates:

  ATR  : keep 3/10/19-condition points (old uncertainty_validation.ipynb source),
         ADD the new 7-condition model (run_eval_ensemble.py: seen 61.67 / unseen 13.38)
         alongside the existing 6-condition point -> x = [3, 6, 7, 10, 19].
  Waymo: 10-condition point updated to the epoch_30 GPT-4o model (= Fig 8b bars):
         day_night 30.62 -> 37.01, dawn_dusk 35.2 -> 41.46. 3/20/30 left on old ckpt.

Keeps the green/red consistency spans + threshold line from the reference.
Lines use magenta/purple (blue is used elsewhere).

Run:  /data/ataparia/conda/envs/gen_ai/bin/python make_conditions_figure.py
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

L1 = "#D81B60"   # series 1 (magenta)
L2 = "#7B1FA2"   # series 2 (purple)


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", stem + ".pdf / .png")


def conditions_plot(stem, counts, s1, s2, lab1, lab2, threshold, ylim, xticks=None,
                    legend_loc="upper right", span_label_y=0.93, peak_x=None):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    x_min, x_max = 0, max(counts) + 1

    # consistency regions (semantic background)
    ax.axvspan(x_min, threshold, color="green", alpha=0.18, zorder=0)
    ax.axvspan(threshold, x_max, color="red",   alpha=0.18, zorder=0)
    ax.axvline(threshold, color="black", alpha=0.40, linestyle="--", linewidth=2, zorder=1)

    # light reference grid
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    # in-region consistency labels (declutters the legend)
    ax.text((x_min + threshold) / 2, ylim[0] + span_label_y * (ylim[1] - ylim[0]),
            "Consistency\n> 95%", ha="center", va="center", fontsize=16,
            fontweight="bold", color="#1B5E20")
    ax.text((threshold + x_max) / 2, ylim[0] + span_label_y * (ylim[1] - ylim[0]),
            "Consistency\n< 95%", ha="center", va="center", fontsize=16,
            fontweight="bold", color="#8E1B1B")

    ax.plot(counts, s1, marker="o", markersize=12, linewidth=3.6, color=L1,
            markeredgecolor="white", markeredgewidth=1.2, label=lab1, zorder=4)
    ax.plot(counts, s2, marker="s", markersize=11, linewidth=3.6, color=L2,
            markeredgecolor="white", markeredgewidth=1.2, label=lab2, zorder=4)

    # star at the peak (optimal #conditions) on both series
    if peak_x is not None and peak_x in counts:
        pi = counts.index(peak_x)
        ax.plot(peak_x, s1[pi], marker="*", markersize=26, color=L1,
                markeredgecolor="white", markeredgewidth=1.2, linestyle="None", zorder=6)
        ax.plot(peak_x, s2[pi], marker="*", markersize=24, color=L2,
                markeredgecolor="white", markeredgewidth=1.2, linestyle="None", zorder=6)

    ax.set_xlabel("Number of Conditions", fontsize=23)
    ax.set_ylabel("mAP", fontsize=23)
    ax.tick_params(axis="both", which="major", labelsize=19)
    ax.set_xticks(xticks if xticks is not None else sorted(set(counts + [threshold])))
    ax.set_ylim(*ylim)
    ax.set_xlim(x_min, x_max)
    ax.legend(fontsize=17, framealpha=0.96, loc=legend_loc, edgecolor="#B8BEC7")
    plt.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------- ATR (Fig 9a)
# 6-cond kept; 7-cond added (seen 61.67 / unseen 13.38, run_eval_ensemble.py)
atr_counts = [3, 6, 7, 10, 19]
atr_seen   = [44.49, 58.75, 61.67, 54.95, 44.41]
atr_unseen = [7.01, 10.02, 13.38, 8.30, 6.72]
conditions_plot("atr_conditions", atr_counts, atr_seen, atr_unseen,
                "Seen Distance", "Unseen Distance", threshold=8,
                ylim=(0.0, 90.0), xticks=[3, 6, 7, 8, 10, 19],
                legend_loc="upper right", span_label_y=0.32, peak_x=7)

# ---------------------------------------------------------------- Waymo (Fig 9b)
# 10-cond updated to epoch_30 (day_night 37.01 / dawn_dusk 41.46); 3/20/30 old
wy_counts    = [3, 10, 20, 30]
wy_day_night = [28.58, 37.01, 29.20, 26.29]
wy_dawn_dusk = [32.37, 41.46, 35.05, 31.83]
conditions_plot("waymo_conditions", wy_counts, wy_day_night, wy_dawn_dusk,
                "Day-Night (Seen)", "Dawn-Dusk (Unseen)", threshold=15,
                ylim=(20.0, 50.0), xticks=[3, 10, 15, 20, 30],
                legend_loc="upper right", span_label_y=0.12, peak_x=10)

print("ALL CONDITION FIGURES DONE ->", OUT)
