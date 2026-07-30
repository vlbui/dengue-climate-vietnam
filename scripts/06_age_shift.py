"""06 - Age-shift sub-analysis: is dengue moving from children to adults?

Uses GBD 2023 dengue Incidence by 5-year age group, both sexes, 1990-2023.
Computes:
  * Mean age of cases (population-weighted across the 17 age bins,
    using midpoints).
  * Share of cases in each broad age band: <15y, 15-29y, 30-49y, 50+y.
  * AAPC of mean age and of adult share.

A rising mean age and a falling <15 share is the classical signature of a
dropping force-of-infection, common in countries where dengue moves from
hyper-endemic (childhood) to mesoendemic (adolescent / adult) dynamics.
This is a side-finding that, if positive, adds a paragraph to the
manuscript discussion under "shifting transmission dynamics".

Outputs:
  tables/age_shift.csv
  figures/static/fig_age_shift.{png,svg}
"""

from __future__ import annotations

import re
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
LANCET_PURPLE = "#925E9F"
LANCET_TEAL = "#0099B4"
LANCET_INK = "#1B1919"

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "processed"
TABLES = PROJ / "tables"
FIG_STATIC = PROJ / "figures" / "static"


def age_midpoint(name: str) -> float:
    """Convert GBD age band names like '5-9 years', '<5 years', '80+ years'
    to numeric midpoints (years). Returns NaN for aggregate labels."""
    name = name.strip()
    if name.startswith("<"):
        m = re.match(r"<(\d+)", name)
        return float(int(m.group(1))) / 2 if m else float("nan")
    if "+" in name:
        m = re.match(r"(\d+)\+", name)
        return float(int(m.group(1)) + 2) if m else float("nan")
    m = re.match(r"(\d+)\s*-\s*(\d+)", name)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a + b) / 2.0
    return float("nan")


def age_band_broad(mid: float) -> str:
    if mid < 15:
        return "<15"
    if mid < 30:
        return "15-29"
    if mid < 50:
        return "30-49"
    return "50+"


def aapc(years, vals):
    m = np.isfinite(vals) & (vals > 0)
    if m.sum() < 5:
        return float("nan")
    slope = np.polyfit(years[m], np.log(vals[m]), 1)[0]
    return (np.exp(slope) - 1.0) * 100.0


def aapc_linear(years, vals):
    """For a value that is itself a number-of-years (mean age), use the
    raw slope rather than log slope."""
    m = np.isfinite(vals)
    if m.sum() < 5:
        return float("nan")
    slope = np.polyfit(years[m], vals[m], 1)[0]
    return slope


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG_STATIC.mkdir(parents=True, exist_ok=True)

    by_age = pd.read_csv(DATA / "gbd_dengue_vn_byage.csv")
    inc = by_age[(by_age.measure_name == "Incidence")
                 & (by_age.metric_name == "Number")].copy()
    inc["midpoint"] = inc["age_name"].map(age_midpoint)
    inc = inc.dropna(subset=["midpoint", "val"])
    inc["broad_band"] = inc["midpoint"].map(age_band_broad)

    # Mean age of cases per year
    yearly = []
    for y, sub in inc.groupby("year"):
        total = sub["val"].sum()
        if total <= 0:
            continue
        mean_age = float((sub["midpoint"] * sub["val"]).sum() / total)
        bands = sub.groupby("broad_band")["val"].sum() / total
        row = dict(year=int(y), total_cases=total, mean_age=mean_age)
        for band in ("<15", "15-29", "30-49", "50+"):
            row[f"share_{band}"] = float(bands.get(band, 0.0))
        yearly.append(row)
    out = pd.DataFrame(yearly).sort_values("year").reset_index(drop=True)
    # restrict to manuscript window
    out = out[(out.year >= 1990) & (out.year <= 2023)].reset_index(drop=True)
    out.to_csv(TABLES / "age_shift.csv", index=False)

    yrs = out["year"].to_numpy()
    slope_meanage = aapc_linear(yrs, out["mean_age"].to_numpy())
    aapc_under15 = aapc(yrs, out["share_<15"].to_numpy())
    aapc_30plus = aapc(yrs, (out["share_30-49"] + out["share_50+"]).to_numpy())

    print(f"  [06] {len(out)} years analyzed ({out.year.min()}-{out.year.max()})")
    print(f"  [06] mean age 1990: {out.mean_age.iloc[0]:.1f}y   "
          f"2023: {out.mean_age.iloc[-1]:.1f}y   "
          f"change: {slope_meanage:+.3f} y/year")
    print(f"  [06] share <15  1990: {out['share_<15'].iloc[0]*100:.1f}%   "
          f"2023: {out['share_<15'].iloc[-1]*100:.1f}%   "
          f"AAPC: {aapc_under15:+.2f}%/yr")
    print(f"  [06] share 30+  1990: "
          f"{(out['share_30-49']+out['share_50+']).iloc[0]*100:.1f}%   "
          f"2023: {(out['share_30-49']+out['share_50+']).iloc[-1]*100:.1f}%   "
          f"AAPC: {aapc_30plus:+.2f}%/yr")

    _build_figure(out, slope_meanage)


def _build_figure(out: pd.DataFrame, slope_meanage: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    yrs = out["year"].to_numpy()

    # Panel A: mean age trend
    ax = axes[0]
    ax.plot(yrs, out["mean_age"], color=LANCET_PURPLE, lw=1.8,
            marker="o", ms=4)
    # OLS line
    if len(yrs) >= 5:
        coef = np.polyfit(yrs, out["mean_age"].to_numpy(), 1)
        ax.plot(yrs, np.polyval(coef, yrs), color=LANCET_INK, lw=1.0,
                ls="--", label=f"OLS slope = {slope_meanage:+.3f} yr/yr")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean age of dengue cases (years)")
    ax.set_title("(A) GBD modelled mean age, Vietnam dengue")
    ax.grid(True, ls="--", alpha=0.3)

    # Panel B: stacked share bands
    ax = axes[1]
    bands = ["<15", "15-29", "30-49", "50+"]
    colors = [LANCET_BLUE, LANCET_GREEN, LANCET_RED, LANCET_PURPLE]
    bottom = np.zeros(len(yrs))
    for band, col in zip(bands, colors):
        share = out[f"share_{band}"].to_numpy() * 100
        ax.fill_between(yrs, bottom, bottom + share, color=col, alpha=0.85,
                        label=f"{band}y")
        bottom = bottom + share
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of cases (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(yrs.min(), yrs.max())
    ax.set_title("(B) Age composition of GBD modelled dengue cases")
    ax.legend(fontsize=9, loc="upper right", frameon=False, ncol=2)
    ax.grid(True, ls="--", alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_STATIC / "fig_age_shift.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_STATIC / "fig_age_shift.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [06] figure: figures/static/fig_age_shift.png + .svg")


def verify():
    out = pd.read_csv(TABLES / "age_shift.csv")
    # share columns should sum to ~1 each year
    band_cols = [c for c in out.columns if c.startswith("share_")]
    sums = out[band_cols].sum(axis=1)
    max_dev = float((sums - 1.0).abs().max())
    print(f"  [verify] max deviation from share-sum=1: {max_dev:.4f}")
    assert max_dev < 0.01, "age-band shares should sum to 1 each year"
    print("  [verify] age-band shares sum to 1 within 1%")


if __name__ == "__main__":
    main()
    verify()
