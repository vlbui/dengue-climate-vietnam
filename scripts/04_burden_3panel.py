"""04 - Three-panel burden figure (Figure 1 expansion for manuscript v3).

Lays out three views of Vietnam's dengue burden 1990-2023 side by side:
  Panel A - Reported cases (OpenDengue)
  Panel B - Reported case rate per 100k (using GBD population denominator)
  Panel C - GBD modelled DALYs (Number, all ages)

The point: if all three trend together, the upward trend is biologically
real. If cases and rate trend up but DALYs do not, the trend is dominated
by surveillance growth + population growth (ascertainment artefact).

Outputs:
  figures/static/fig1_burden_3panel.{png,svg}
  tables/burden_3panel_data.csv
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
LANCET_GREEN = "#42B540"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "processed"
TABLES = PROJ / "tables"
FIG_STATIC = PROJ / "figures" / "static"
POP_FILE = _REPO_ROOT / "raw_data" / "burden" / "query4_pop.csv"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG_STATIC.mkdir(parents=True, exist_ok=True)

    od = pd.read_csv(DATA / "opendengue_vn_annual.csv")
    gbd = pd.read_csv(DATA / "gbd_dengue_vn.csv")
    pop = pd.read_csv(POP_FILE)

    pop_b = (pop[(pop.sex_name == "Both")
                 & (pop.age_name == "All ages")
                 & (pop.metric_name == "Number")]
             [["year", "val"]]
             .rename(columns={"val": "population"}))
    daly = (gbd[(gbd.measure_name == "DALYs (Disability-Adjusted Life Years)")
                & (gbd.metric_name == "Number")
                & (gbd.age_name == "All ages")]
            [["year", "val", "lower", "upper"]]
            .rename(columns={"val": "daly", "lower": "daly_lo",
                             "upper": "daly_hi"}))

    df = (od[["year", "cases", "source_label"]]
          .merge(pop_b, on="year", how="left")
          .merge(daly, on="year", how="left"))
    df["rate_per_100k"] = df["cases"] / df["population"] * 1e5
    df.to_csv(TABLES / "burden_3panel_data.csv", index=False)

    print(f"  [04] paired rows: {len(df)} years "
          f"({df.year.min()}-{df.year.max()})")
    print(f"  [04] rate per 100k 1990: {df.loc[df.year==1990,'rate_per_100k'].iloc[0]:.1f}  "
          f"2022: {df.loc[df.year==2022,'rate_per_100k'].iloc[0]:.1f}")
    if df["daly"].notna().any():
        d90 = df.loc[df.year == 1990, "daly"].iloc[0]
        d22 = df.loc[df.year == 2022, "daly"].iloc[0]
        print(f"  [04] GBD DALYs    1990: {d90:>10,.0f}  2022: {d22:>10,.0f}")

    _build_figure(df)


def _add_aapc_label(ax, year, val, text, color):
    # Place the label directly above the marked point. Keeping it centred
    # on the point (rather than offset to the right) guarantees the text
    # stays inside the plot margins even for the right-most peak year.
    ax.annotate(text, xy=(year, val), xytext=(0, 9),
                textcoords="offset points", fontsize=9,
                color=color, ha="center", va="bottom", weight="bold")


def _aapc(years, vals):
    """Fit log-linear, return annual % change."""
    m = np.isfinite(vals) & (vals > 0)
    if m.sum() < 5:
        return float("nan")
    slope = np.polyfit(years[m], np.log(vals[m]), 1)[0]
    return (np.exp(slope) - 1.0) * 100.0


def _build_figure(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)
    yrs = df["year"].to_numpy()

    # ---- Panel A: cases (linear) with peak annotations ----
    ax = axes[0]
    ax.plot(yrs, df["cases"], color=LANCET_RED, lw=1.8, marker="o", ms=4)
    aapc_a = _aapc(yrs, df["cases"].to_numpy())
    for y in (1998, 2022):
        if y in df["year"].values:
            v = df.loc[df.year == y, "cases"].iloc[0]
            ax.scatter([y], [v], s=80, edgecolor=LANCET_INK,
                       facecolor=LANCET_RED, lw=1.2, zorder=5)
            _add_aapc_label(ax, y, v, f"{int(v):,}", LANCET_INK)
    ax.set_ylabel("Reported cases (Number)")
    ax.set_title(f"(A) OpenDengue cases  (AAPC = {aapc_a:+.1f}%/yr)")
    ax.grid(True, ls="--", alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}")
    )

    # ---- Panel B: rate per 100k ----
    ax = axes[1]
    ax.plot(yrs, df["rate_per_100k"], color=LANCET_GREEN, lw=1.8, marker="o", ms=4)
    aapc_b = _aapc(yrs, df["rate_per_100k"].to_numpy())
    for y in (1998, 2022):
        if y in df["year"].values:
            v = df.loc[df.year == y, "rate_per_100k"].iloc[0]
            ax.scatter([y], [v], s=80, edgecolor=LANCET_INK,
                       facecolor=LANCET_GREEN, lw=1.2, zorder=5)
            _add_aapc_label(ax, y, v, f"{v:.0f}", LANCET_INK)
    ax.set_ylabel("Reported case rate per 100,000")
    ax.set_title(f"(B) Cases / GBD pop  (AAPC = {aapc_b:+.1f}%/yr)")
    ax.grid(True, ls="--", alpha=0.3)

    # ---- Panel C: GBD DALYs (Number, all ages) ----
    ax = axes[2]
    sub = df.dropna(subset=["daly"])
    ax.plot(sub.year, sub["daly"], color=LANCET_BLUE, lw=1.8, marker="o", ms=4,
            label="GBD DALYs (Number)")
    ax.fill_between(sub.year, sub["daly_lo"], sub["daly_hi"],
                    color=LANCET_BLUE, alpha=0.15, lw=0,
                    label="95% UI")
    aapc_c = _aapc(sub.year.to_numpy(), sub["daly"].to_numpy())
    for y in (1998, 2022):
        if y in sub["year"].values:
            v = sub.loc[sub.year == y, "daly"].iloc[0]
            ax.scatter([y], [v], s=80, edgecolor=LANCET_INK,
                       facecolor=LANCET_BLUE, lw=1.2, zorder=5)
            _add_aapc_label(ax, y, v, f"{int(v):,}", LANCET_INK)
    ax.set_ylabel("DALYs (Number)")
    ax.set_xlabel("Year")
    ax.set_title(f"(C) GBD modelled DALYs  (AAPC = {aapc_c:+.1f}%/yr)")
    # No legend: the panel title names the series and the shaded band is
    # explained in the figure caption. A legend here concealed a data point.
    ax.grid(True, ls="--", alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}")
    )

    for a in axes:
        a.set_xlabel("Year")
        # Extra vertical headroom so the above-point labels stay inside
        # the axes even at the tallest peak.
        a.margins(y=0.14)

    fig.tight_layout()
    fig.savefig(FIG_STATIC / "fig1_burden_3panel.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_STATIC / "fig1_burden_3panel.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [04] figure: figures/static/fig1_burden_3panel.png + .svg")
    print(f"  [04] AAPC trio:  cases={aapc_a:+.1f}%/yr  rate={aapc_b:+.1f}%/yr  "
          f"DALYs={aapc_c:+.1f}%/yr")
    print(f"  [04] -> if rate AAPC > DALYs AAPC, much of the cases trend is "
          "surveillance growth")


def verify():
    df = pd.read_csv(TABLES / "burden_3panel_data.csv")
    n = len(df)
    n_with_pop = df["population"].notna().sum()
    n_with_daly = df["daly"].notna().sum()
    print(f"  [verify] {n} years; pop coverage {n_with_pop}/{n}; "
          f"DALY coverage {n_with_daly}/{n}")
    assert n_with_pop >= 30, "missing population denominator coverage"
    assert n_with_daly >= 30, "missing GBD DALY coverage"
    print("  [verify] both denominators cover >= 30 years")


if __name__ == "__main__":
    main()
    verify()
