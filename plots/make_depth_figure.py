#!/usr/bin/env python
"""
ATR VLC-block depth ablation figure: number of VLC blocks vs. mAP (seen + unseen).
Points: 1, 2, 4, 6 blocks (8-block point omitted). All n=7 conditions, COCO mAP@[.5:.95],
run_eval_ensemble.py. Numbers are TEST mAP in %.

  1 block : seen 59.74 / unseen 14.12
  2 block : seen 61.67 / unseen 13.38
  4 block : seen 61.41 / unseen 12.74
  6 block : seen 59.70 / unseen 13.70

Run:  /data/ataparia/conda/envs/gen_ai/bin/python make_depth_figure.py
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

VLC  = "#D81B60"   # seen   (magenta)
BASE = "#7B1FA2"   # unseen (purple — matches the conditions plot)
GRID = "#B8BEC7"

blocks      = [1, 2, 4, 6]
seen_map    = [59.74, 61.67, 61.41, 59.70]
unseen_map  = [14.12, 13.38, 12.74, 13.70]

fig, ax = plt.subplots(figsize=(7.2, 3.8))

# subtle guide marking the chosen depth (2 blocks), tied to the axis
ax.axvline(x=2, color=GRID, linewidth=1.3, linestyle=(0, (4, 4)), alpha=0.8, zorder=1)

ax.plot(blocks, seen_map, color=VLC, linewidth=2.8, marker="o", markersize=10,
        markeredgecolor="white", markeredgewidth=1.1, label="Seen Distance", zorder=4)
ax.plot(blocks, unseen_map, color=BASE, linewidth=2.6, marker="s", markersize=9,
        markeredgecolor="white", markeredgewidth=1.1, label="Unseen Distance", zorder=4)

# highlight the chosen default (2 VLC blocks): same-colour stars, sized to hide
# the underlying round/square markers
ax.plot(2, seen_map[1], marker="*", markersize=22, color=VLC,
        markeredgecolor="white", markeredgewidth=1.0, linestyle="None", zorder=6)
ax.plot(2, unseen_map[1], marker="*", markersize=22, color=BASE,
        markeredgecolor="white", markeredgewidth=1.0, linestyle="None", zorder=6)

# value labels
for x, y in zip(blocks, seen_map):
    dy = 14 if x == 2 else 11           # extra clearance above the star
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=13, color=VLC, fontweight="bold")
for x, y in zip(blocks, unseen_map):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=13, color=BASE, fontweight="bold")

ax.set_xticks(blocks)
ax.set_xticklabels(["1", "2 (Ours)", "4", "6"], fontsize=16)
# bold + black the chosen "2 (Ours)" tick, right along the axis
for lbl, b in zip(ax.get_xticklabels(), blocks):
    if b == 2:
        lbl.set_fontweight("bold")
        lbl.set_color("black")
ax.set_xlabel("Number of VLC blocks", fontsize=18, labelpad=14)
ax.set_ylabel("mAP", fontsize=18)
ax.tick_params(axis="y", labelsize=14)
ax.set_ylim(0, 70)
ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=15, loc="center right", framealpha=0.95, edgecolor=GRID)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"atr_depth_ablation.{ext}"))
plt.close(fig)
print("wrote atr_depth_ablation.pdf / .png ->", OUT)
