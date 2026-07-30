"""12 - Supplementary Figure S2: period-stratified robustness.

Bars show lag-0 Spearman rho between annual dengue cases and each of the
five top primary predictors within three EQUAL 12-year periods (revised
following peer review, which asked for equal-length windows rather than the
earlier uneven spans):
  Era A 1990-2001
  Era B 2002-2013
  Era C 2014-2025

Predictors: AMO (ann), IOBW (spr), AMO (spr), IOBW (ann), TNA (ann).
AMO ends in 2022 in the climate index file, so AMO bars in Era C rest on the
paired years AMO covers (paired N logged to stdout).

Reads : final/data_inputs/opendengue_vn_annual.csv
        final/data_inputs/climate_indices.csv
Writes: final/figures/FigS2_era_stratified.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr as _spearmanr

    def spearman(a, b):
        return _spearmanr(a, b)[0]
except Exception:  # scipy not available -> rank-based Pearson (== Spearman, no ties)
    def spearman(a, b):
        ar = pd.Series(np.asarray(a, float)).rank()
        br = pd.Series(np.asarray(b, float)).rank()
        return float(np.corrcoef(ar, br)[0, 1])

LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_GREEN = "#42B540"
LANCET_INK = "#1B1919"

FINAL = Path(__file__).resolve().parents[1]
DENGUE = FINAL / "data_inputs" / "opendengue_vn_annual.csv"
CLIMATE = FINAL / "data_inputs" / "climate_indices.csv"
OUT = FINAL / "figures"

PREDICTORS = [
    ("AMO", "annual", "AMO (ann)"),
    ("IOBW", "MAM", "IOBW (spr)"),
    ("AMO", "MAM", "AMO (spr)"),
    ("IOBW", "annual", "IOBW (ann)"),
    ("TNA", "annual", "TNA (ann)"),
]
ERAS = [
    ("Era A (1990-2001)", 1990, 2001, LANCET_BLUE),
    ("Era B (2002-2013)", 2002, 2013, LANCET_GREEN),
    ("Era C (2014-2025)", 2014, 2025, LANCET_RED),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    od = pd.read_csv(DENGUE)[["year", "cases"]]
    ci = pd.read_csv(CLIMATE)

    rho = np.full((len(PREDICTORS), len(ERAS)), np.nan)
    for i, (idx, season, _label) in enumerate(PREDICTORS):
        c = ci[(ci.index_name == idx) & (ci.season == season)][["year", "value"]]
        merged = od.merge(c, on="year", how="inner")
        for j, (era, lo, hi, _col) in enumerate(ERAS):
            e = merged[(merged.year >= lo) & (merged.year <= hi)]
            if len(e) >= 3:
                rho[i, j] = spearman(e["value"], e["cases"])
            print(f"  [S2] {idx:5} {season:6} {era}: paired N={len(e):2d}, "
                  f"rho={rho[i, j]:+.3f}")

    labels = [lbl for _, _, lbl in PREDICTORS]
    x = np.arange(len(PREDICTORS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for j, (era_label, _lo, _hi, color) in enumerate(ERAS):
        ax.bar(x + (j - 1) * w, rho[:, j], width=w, color=color,
               edgecolor=LANCET_INK, linewidth=0.5, label=era_label, zorder=3)
    ax.axhline(0, color=LANCET_INK, lw=1.0, zorder=4)
    ax.set_ylim(-0.15, 0.70)
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_ylabel("Spearman rho (lag 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(PREDICTORS) - 0.4)
    ax.grid(True, ls="--", alpha=0.3, axis="y", zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "FigS2_era_stratified.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "FigS2_era_stratified.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [S2] figure: {OUT / 'FigS2_era_stratified.png'} + .svg")


if __name__ == "__main__":
    main()
