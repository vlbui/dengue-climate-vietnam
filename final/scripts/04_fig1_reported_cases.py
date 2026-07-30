"""04 - Figure 1: Vietnam annual reported dengue cases, 1990-2025.

Single-panel version (replaces the earlier three-panel burden figure).
Following peer review, the rate-per-100k and GBD modelled-DALY panels were
dropped from the analysis; Figure 1 now shows only the reported-case series
with a fitted log-linear trend and the 1998 / 2022 outbreak peaks marked.

Reads : final/data_inputs/opendengue_vn_annual.csv
Writes: final/figures/Fig1_reported_cases.{png,svg}
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
from matplotlib.ticker import FuncFormatter

LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

FINAL = Path(__file__).resolve().parents[1]
DENGUE = FINAL / "data_inputs" / "opendengue_vn_annual.csv"
OUT = FINAL / "figures"

# Outbreak peaks to annotate (year, cases)
PEAKS = [(1998, 234866), (2022, 367729)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    od = pd.read_csv(DENGUE).sort_values("year")
    yr = od["year"].to_numpy(dtype=float)
    cases = od["cases"].to_numpy(dtype=float)

    # Fitted log-linear trend (OLS on log cases) -> exponential on linear axis.
    slope, intercept = np.polyfit(yr, np.log(cases), 1)
    trend = np.exp(intercept + slope * yr)
    pct_per_year = (np.exp(slope) - 1) * 100
    print(f"  [04] log-linear slope = {slope:.4f}/yr  ->  +{pct_per_year:.2f}% per year")

    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.plot(yr, cases, "-", color=LANCET_BLUE, lw=1.6, zorder=3)
    ax.plot(yr, cases, "o", color=LANCET_BLUE, ms=4, zorder=4)
    ax.plot(yr, trend, "--", color=LANCET_GREY, lw=1.8, zorder=2,
            label=f"Fitted log-linear trend (+{pct_per_year:.1f}%/year)")

    for py, pv in PEAKS:
        ax.plot(py, pv, "o", color=LANCET_RED, ms=8, zorder=5)
        ax.annotate(f"{py}\n({pv:,})", (py, pv), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9,
                    color=LANCET_RED, fontweight="bold")

    ax.set_ylabel("Reported dengue cases")
    ax.set_xlabel("Year")
    ax.set_xlim(1989, 2026)
    ax.set_ylim(0, 410000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, ls="--", alpha=0.3, axis="y", zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "Fig1_reported_cases.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Fig1_reported_cases.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [04] figure: {OUT / 'Fig1_reported_cases.png'} + .svg")


if __name__ == "__main__":
    main()
