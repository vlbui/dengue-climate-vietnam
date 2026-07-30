"""Supplementary Figure S4 - effect-size stability for spring IOBW.

For each of 13 specifications, fit an ordinary least squares (OLS) line
from spring IOBW (or annual IOBW where stated) to ln(reported cases) and
report the slope with a 5,000-resample paired bootstrap 95 percent CI.

Specifications, grouped:
  Core         Primary (raw, lag 0)
               Lag 1 (climate -> next-year cases)
               First-difference
               Linear-detrended
               Quadratic-detrended
  Aggregation  Annual IOBW (raw)
  Sensitivity  Exclude 2011
               Exclude COVID (2020-22)
               Exclude pre-1995
  Window       Window 1990-2015
               Window 1995-2020
               Window 2000-2025
               Window 2005-2025

Closed circle: CI excludes zero.  Open circle: CI crosses zero.

Output:
  manuscript/v4/figures/supplement/S_Figure_S4_iobw_slope_specs.{png,svg}
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
LANCET_TEAL = "#0099B4"
PURPLE = "#925E9F"

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "manuscript" / "v4" / "figures" / "supplement"
DENGUE = PROJ / "data" / "processed" / "opendengue_vn_annual.csv"
CLIMATE = _REPO_ROOT / "raw_data" / "climate" / "climate_indices.csv"

N_BOOT = 5000
SEED = 20260524

GROUPS = {
    "Core":        LANCET_RED,
    "Aggregation": LANCET_BLUE,
    "Sensitivity": LANCET_GREEN,
    "Window":      PURPLE,
}


def linear_detrend(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def quadratic_detrend(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    A = np.vstack([x ** 2, x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                    seed: int = SEED) -> tuple[float, float, float]:
    """Paired bootstrap: returns (point slope, CI lo, CI hi) at 95%."""
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        slopes[i] = np.polyfit(x[idx], y[idx], 1)[0]
    point = float(np.polyfit(x, y, 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    od = pd.read_csv(DENGUE)[["year", "cases"]].sort_values("year").reset_index(drop=True)
    ci = pd.read_csv(CLIMATE)
    iobw_mam = (ci[(ci.index_name == "IOBW") & (ci.season == "MAM")]
                [["year", "value"]].rename(columns={"value": "iobw_spr"}))
    iobw_ann = (ci[(ci.index_name == "IOBW") & (ci.season == "annual")]
                [["year", "value"]].rename(columns={"value": "iobw_ann"}))
    df = od.merge(iobw_mam, on="year").merge(iobw_ann, on="year")
    df["ln_cases"] = np.log(df["cases"])
    df = df.sort_values("year").reset_index(drop=True)

    specs: list[dict] = []

    def add(label, group, x, y):
        point, lo, hi = bootstrap_slope(np.asarray(x, float), np.asarray(y, float))
        specs.append(dict(label=label, group=group, n=len(x),
                          point=point, lo=lo, hi=hi))

    # ---- Core ----
    add("Primary (raw, lag 0)", "Core",
        df.iobw_spr, df.ln_cases)
    # Lag 1: climate at year t predicts cases at year t+1
    add("Lag 1 (climate -> next-year cases)", "Core",
        df.iobw_spr.iloc[:-1], df.ln_cases.iloc[1:])
    add("First-difference", "Core",
        np.diff(df.iobw_spr.to_numpy()),
        np.diff(df.ln_cases.to_numpy()))
    add("Linear-detrended", "Core",
        linear_detrend(df.iobw_spr.to_numpy()),
        linear_detrend(df.ln_cases.to_numpy()))
    add("Quadratic-detrended", "Core",
        quadratic_detrend(df.iobw_spr.to_numpy()),
        quadratic_detrend(df.ln_cases.to_numpy()))

    # ---- Aggregation ----
    add("Annual IOBW (raw)", "Aggregation",
        df.iobw_ann, df.ln_cases)

    # ---- Sensitivity ----
    no11 = df[df.year != 2011]
    add("Exclude 2011", "Sensitivity", no11.iobw_spr, no11.ln_cases)
    noCovid = df[~df.year.isin([2020, 2021, 2022])]
    add("Exclude COVID (2020-22)", "Sensitivity",
        noCovid.iobw_spr, noCovid.ln_cases)
    no95 = df[df.year >= 1995]
    add("Exclude pre-1995", "Sensitivity", no95.iobw_spr, no95.ln_cases)

    # ---- Window ----
    w1 = df[(df.year >= 1990) & (df.year <= 2015)]
    add("Window 1990-2015", "Window", w1.iobw_spr, w1.ln_cases)
    w2 = df[(df.year >= 1995) & (df.year <= 2020)]
    add("Window 1995-2020", "Window", w2.iobw_spr, w2.ln_cases)
    w3 = df[(df.year >= 2000) & (df.year <= 2025)]
    add("Window 2000-2025", "Window", w3.iobw_spr, w3.ln_cases)
    w4 = df[(df.year >= 2005) & (df.year <= 2025)]
    add("Window 2005-2025", "Window", w4.iobw_spr, w4.ln_cases)

    for s in specs:
        sign = "CI > 0" if s["lo"] > 0 else ("CI < 0" if s["hi"] < 0
                                              else "crosses 0")
        print(f"  [S4] {s['label']:40} slope={s['point']:+.3f} "
              f"[{s['lo']:+.3f}, {s['hi']:+.3f}]  n={s['n']:2d}  {sign}")
    n_excl_zero = sum(1 for s in specs if s["lo"] > 0 or s["hi"] < 0)
    print(f"  [S4] {n_excl_zero}/{len(specs)} specifications have a CI that "
          "excludes zero")

    _build_figure(specs, n_excl_zero)


def _build_figure(specs: list[dict], n_excl_zero: int) -> None:
    n = len(specs)
    # Top of plot at y=0, bottom at y=n-1 (reverse so first row is on top).
    y_pos = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(12, 6.2))

    for y, s in zip(y_pos, specs):
        color = GROUPS[s["group"]]
        excludes_zero = s["lo"] > 0 or s["hi"] < 0
        face = color if excludes_zero else "white"
        ax.plot([s["lo"], s["hi"]], [y, y], color=color, lw=2.0, zorder=3,
                solid_capstyle="round", alpha=0.85)
        ax.scatter([s["point"]], [y], s=70, facecolor=face,
                   edgecolor=color, lw=1.4, zorder=5)
        ax.text(1.005, y, f"n={s['n']}", transform=ax.get_yaxis_transform(),
                fontsize=7.5, color=LANCET_GREY, ha="left", va="center")

    ax.axvline(0, color=LANCET_INK, ls="--", lw=1.0, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s["label"] for s in specs], fontsize=9)
    ax.set_xlabel("Bootstrap OLS slope (spring IOBW -> ln cases) with 95% CI")
    ax.set_xlim(-0.7, 2.2)
    ax.grid(True, ls="--", alpha=0.3, axis="x")

    # Legend entries: group colours, plus filled / open marker convention.
    from matplotlib.lines import Line2D
    grp_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=col,
               markeredgecolor=col, markersize=8, label=name)
        for name, col in GROUPS.items()
    ]
    grp_handles.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=LANCET_INK, markersize=8,
               label="CI crosses zero (open)")
    )
    ax.legend(handles=grp_handles, loc="lower right", fontsize=8,
              frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT / "S_Figure_S4_iobw_slope_specs.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "S_Figure_S4_iobw_slope_specs.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [S4] {OUT / 'S_Figure_S4_iobw_slope_specs.png'} + .svg")


if __name__ == "__main__":
    main()
