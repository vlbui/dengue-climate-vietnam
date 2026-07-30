"""Figure 5 - standardized time series of dengue cases and leading climate
indices, 1990 to 2025.

Replaces the previous Figure 5 (age shift of GBD-modelled cases), which has
been removed from the manuscript following coauthor review.

Each series is z-scored over the analysis window (1990-2025) so that a
common ordinate lets the reader inspect co-movement and outbreak alignment
directly. Spring (MAM) IOBW and the annual AMO are the two leading
climate predictors identified in the raw and first-differenced lenses
respectively.

A companion supplementary figure (S_Figure_S7_standardized_all_indices)
shows all eight tropical SST indices on the same standardized scale.

Outputs:
  manuscript/v4/figures/Figure_5_standardized_timeseries.{png,svg}
  manuscript/v4/figures/supplement/S_Figure_S7_standardized_all_indices.{png,svg}
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
LANCET_GREEN = "#42B540"
LANCET_ORANGE = "#FDAF17"
LANCET_PURPLE = "#925E9F"
LANCET_TEAL = "#0099B4"
LANCET_BROWN = "#AD002A"

PROJ = Path(__file__).resolve().parents[1]
OUT_MAIN = PROJ / "manuscript" / "v4" / "figures"
OUT_SUPP = PROJ / "manuscript" / "v4" / "figures" / "supplement"
DENGUE = PROJ / "data" / "processed" / "opendengue_vn_annual.csv"
CLIMATE = _REPO_ROOT / "raw_data" / "climate" / "climate_indices.csv"

YEAR_LO, YEAR_HI = 1990, 2025
OUTBREAK_YEARS = [1998, 2022]  # detrended log-case z > +1.5 SD, see _fig_supp_S3


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score over the analysis window (mean 0, SD 1)."""
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def load_series() -> pd.DataFrame:
    """Return a wide DataFrame indexed by year with ln_cases and each
    (index_name, season) pair as a column.
    """
    od = pd.read_csv(DENGUE)[["year", "cases"]]
    od = od[(od.year >= YEAR_LO) & (od.year <= YEAR_HI)].copy()
    od["ln_cases"] = np.log(od["cases"])

    ci = pd.read_csv(CLIMATE)
    ci = ci[(ci.year >= YEAR_LO) & (ci.year <= YEAR_HI)].copy()
    ci["col"] = ci.index_name + "_" + ci.season
    wide = ci.pivot(index="year", columns="col", values="value").reset_index()

    df = od.merge(wide, on="year", how="inner").sort_values("year")
    return df


def main_figure(df: pd.DataFrame) -> None:
    """Two-panel main-text figure: leading raw predictor (spring IOBW) and
    leading first-differenced predictor (annual AMO), each plotted against
    ln(reported cases) on a z-score axis.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(10.0, 7.0), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    cfg = [
        {
            "ax": axes[0],
            "clim_col": "IOBW_MAM",
            "clim_label": "Spring (MAM) IOBW SST anomaly",
            "panel": "(A)",
            "subtitle": "Leading raw predictor: spring IOBW",
        },
        {
            "ax": axes[1],
            "clim_col": "AMO_annual",
            "clim_label": "Annual AMO SST anomaly",
            "panel": "(B)",
            "subtitle": "Leading first-differenced predictor: annual AMO",
        },
    ]

    z_cases = zscore(df["ln_cases"].to_numpy())

    for c in cfg:
        ax = c["ax"]
        z_clim = zscore(df[c["clim_col"]].to_numpy())

        # Outbreak shading
        for yr in OUTBREAK_YEARS:
            ax.axvspan(yr - 0.4, yr + 0.4, color=LANCET_GREY, alpha=0.18,
                       zorder=0)

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
        ax.set_title(f"{c['panel']} {c['subtitle']}",
                     loc="left", fontsize=10, color=LANCET_INK)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=1)

        # Annotate outbreak years
        for yr in OUTBREAK_YEARS:
            yc = z_cases[df.year.to_numpy() == yr][0]
            ax.annotate(str(yr), xy=(yr, yc), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8, color=LANCET_INK)

    axes[1].set_xlabel("Year", fontsize=10, color=LANCET_INK)
    axes[1].set_xlim(YEAR_LO - 0.5, YEAR_HI + 0.5)
    axes[1].set_xticks(np.arange(YEAR_LO, YEAR_HI + 1, 5))

    fig.tight_layout()

    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    png = OUT_MAIN / "Figure_5_standardized_timeseries.png"
    svg = OUT_MAIN / "Figure_5_standardized_timeseries.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Fig5] wrote {png.name} / {svg.name}")


def supp_figure(df: pd.DataFrame) -> None:
    """4x2 grid: ln(cases) plotted with each of the 8 climate indices,
    on a standardized scale. Uses annual mean of the climate index for
    consistency across panels.
    """
    indices = ["IOBW", "AMO", "DMI", "TNA", "MEI", "ONI", "SOI", "PDO"]
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 11.5), sharex=True,
                              sharey=True)
    z_cases = zscore(df["ln_cases"].to_numpy())

    for ax, name, letter in zip(axes.flat, indices, "ABCDEFGH"):
        col = f"{name}_annual"
        if col not in df.columns:
            continue
        z_clim = zscore(df[col].to_numpy())

        for yr in OUTBREAK_YEARS:
            ax.axvspan(yr - 0.4, yr + 0.4, color=LANCET_GREY, alpha=0.18,
                       zorder=0)

        ax.axhline(0, color=LANCET_GREY, lw=0.6, zorder=1)
        ax.plot(df.year, z_cases, "-", color=LANCET_BLUE, lw=1.4,
                label="ln(cases)", zorder=4)
        ax.plot(df.year, z_clim, "--", color=LANCET_RED, lw=1.4,
                label=f"{name}", zorder=3)

        ax.set_title(f"({letter}) Annual {name}", loc="left", fontsize=9.5,
                     color=LANCET_INK)
        ax.set_ylim(-3.2, 3.2)
        ax.grid(True, ls="--", alpha=0.3)
        if letter == "A":
            ax.legend(loc="upper left", fontsize=8, frameon=False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Standardized", fontsize=9, color=LANCET_INK)
    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=9, color=LANCET_INK)
        ax.set_xticks(np.arange(YEAR_LO, YEAR_HI + 1, 5))

    fig.suptitle(
        "Vietnam dengue cases versus all eight tropical SST indices, "
        f"{YEAR_LO}-{YEAR_HI} (annual means, standardized)",
        fontsize=11.5, color=LANCET_INK, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    png = OUT_SUPP / "S_Figure_S7_standardized_all_indices.png"
    svg = OUT_SUPP / "S_Figure_S7_standardized_all_indices.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SuppS7] wrote {png.name} / {svg.name}")


def main() -> None:
    df = load_series()
    print(f"  [Fig5] loaded {len(df)} years, {len(df.columns)-2} climate cols")
    main_figure(df)
    supp_figure(df)
    print("  [Fig5] done.")


if __name__ == "__main__":
    main()
