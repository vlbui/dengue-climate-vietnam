"""Supplementary Figure S1 - climate predictor panel, 1990 to 2025.

Annual mean (solid blue with circles) and March-May spring mean (dashed
red with squares) of the eight tropical SST indices used as candidate
predictors. Reads the aggregated CSV produced by _climate_to_csv.py.

Output:
  manuscript/v4/figures/supplement/S_Figure_S1_climate_panel.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "manuscript" / "v4" / "figures" / "supplement"
CLIMATE = _REPO_ROOT / "raw_data" / "climate" / "climate_indices.csv"

# Order matches the v2 figure layout (4 rows x 2 cols).
PANEL_ORDER = [
    ("IOBW", "(A)"), ("MEI", "(B)"),
    ("DMI",  "(C)"), ("SOI", "(D)"),
    ("PDO",  "(E)"), ("ONI", "(F)"),
    ("AMO",  "(G)"), ("TNA", "(H)"),
]
YEAR_LO, YEAR_HI = 1990, 2025


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ci = pd.read_csv(CLIMATE)
    ci = ci[(ci.year >= YEAR_LO) & (ci.year <= YEAR_HI)]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for ax, (idx, panel_tag) in zip(axes, PANEL_ORDER):
        ann = (ci[(ci.index_name == idx) & (ci.season == "annual")]
               .sort_values("year"))
        mam = (ci[(ci.index_name == idx) & (ci.season == "MAM")]
               .sort_values("year"))
        ax.axhline(0, color=LANCET_GREY, lw=0.8, zorder=1)
        ax.plot(ann.year, ann.value, "-o", color=LANCET_BLUE, lw=1.4, ms=3.5,
                label=f"{idx} annual", zorder=3)
        ax.plot(mam.year, mam.value, "--s", color=LANCET_RED, lw=1.2, ms=3.2,
                label=f"{idx} spring (MAM)", zorder=3)
        ax.set_title(f"{panel_tag} {idx}", loc="left", fontsize=9.5,
                     color=LANCET_INK)
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7.5, frameon=False)

    for ax in axes[-2:]:
        ax.set_xlabel("Year")

    fig.tight_layout()
    fig.savefig(OUT / "S_Figure_S1_climate_panel.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "S_Figure_S1_climate_panel.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [S1] {OUT / 'S_Figure_S1_climate_panel.png'} + .svg")


if __name__ == "__main__":
    main()
