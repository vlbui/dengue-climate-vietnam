"""06 - Figure 5: standardized time series of dengue cases and the two
leading climate indices, 1990 to 2025.

Replaces the previous Figure 5 (GBD age shift), removed following co-author
review. Each series is z-scored over the analysis window so a common ordinate
lets the reader inspect co-movement and outbreak alignment directly. Spring
(MAM) IOBW is the leading raw predictor; annual AMO is the leading
first-differenced predictor.

Reads : final/data_inputs/opendengue_vn_annual.csv
        final/data_inputs/climate_indices.csv
Writes: final/figures/Fig5_standardized_timeseries.{png,svg}
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

LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

FINAL = Path(__file__).resolve().parents[1]
DENGUE = FINAL / "data_inputs" / "opendengue_vn_annual.csv"
CLIMATE = FINAL / "data_inputs" / "climate_indices.csv"
OUT = FINAL / "figures"

YEAR_LO, YEAR_HI = 1990, 2025
OUTBREAK_YEARS = [1998, 2022]


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def load_series():
    od = pd.read_csv(DENGUE)[["year", "cases"]]
    od = od[(od.year >= YEAR_LO) & (od.year <= YEAR_HI)].copy()
    od["ln_cases"] = np.log(od["cases"])
    ci = pd.read_csv(CLIMATE)
    ci = ci[(ci.year >= YEAR_LO) & (ci.year <= YEAR_HI)].copy()
    ci["col"] = ci.index_name + "_" + ci.season
    wide = ci.pivot(index="year", columns="col", values="value").reset_index()
    return od.merge(wide, on="year", how="inner").sort_values("year")


def main():
    df = load_series()
    print(f"  [Fig5] loaded {len(df)} years")
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    cfg = [
        {"ax": axes[0], "clim_col": "IOBW_MAM",
         "clim_label": "Spring (MAM) IOBW SST anomaly", "panel": "(A)",
         "subtitle": "Leading raw predictor: spring IOBW"},
        {"ax": axes[1], "clim_col": "AMO_annual",
         "clim_label": "Annual AMO SST anomaly", "panel": "(B)",
         "subtitle": "Leading first-differenced predictor: annual AMO"},
    ]
    z_cases = zscore(df["ln_cases"].to_numpy())
    for c in cfg:
        ax = c["ax"]
        z_clim = zscore(df[c["clim_col"]].to_numpy())
        for yr in OUTBREAK_YEARS:
            ax.axvspan(yr - 0.5, yr + 0.5, facecolor="#9AA3A8", alpha=0.38, zorder=0, lw=0)
            ax.axvspan(yr - 0.5, yr + 0.5, facecolor="none", edgecolor="#6E777C",
                       lw=0.8, ls=":", alpha=0.8, zorder=0.5)
        ax.axhline(0, color=LANCET_GREY, lw=0.7, zorder=1)
        ax.plot(df.year, z_cases, "-o", color=LANCET_BLUE, lw=1.8, ms=4.2,
                mfc="white", mec=LANCET_BLUE,
                label="ln(reported cases), z-scored", zorder=4)
        ax.plot(df.year, z_clim, "--s", color=LANCET_RED, lw=1.6, ms=3.8,
                mfc="white", mec=LANCET_RED,
                label=f"{c['clim_label']}, z-scored", zorder=3)
        ax.set_ylabel("Standardized (z-score)", fontsize=10, color=LANCET_INK)
        ax.set_ylim(-3.2, 3.2)
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_title(f"{c['panel']} {c['subtitle']}", loc="left",
                     fontsize=10, color=LANCET_INK)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=1)
        for yr in OUTBREAK_YEARS:
            ax.text(yr, -2.95, str(yr), ha="center", va="bottom",
                    fontsize=8, color="#4A4A4A", fontweight="bold")
    axes[1].set_xlabel("Year", fontsize=10, color=LANCET_INK)
    axes[1].set_xlim(YEAR_LO - 0.5, YEAR_HI + 0.5)
    axes[1].set_xticks(np.arange(YEAR_LO, YEAR_HI + 1, 5))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Fig5_standardized_timeseries.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Fig5_standardized_timeseries.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Fig5] figure: {OUT / 'Fig5_standardized_timeseries.png'} + .svg")


if __name__ == "__main__":
    main()
