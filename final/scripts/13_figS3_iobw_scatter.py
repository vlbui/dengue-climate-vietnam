"""Supplementary Figure S3 - IOBW-focused two-panel scatter.

Panel A: Raw spring IOBW versus log-transformed reported cases (N = 36
         paired years, 1990-2025).
Panel B: First-difference scatter -- delta spring IOBW versus delta log
         cases (N = 35 first differences).

Outbreak-year markers (red): the two outbreak years defined in main-text
Methods and Figure 2 -- detrended log-case z-score above +1.5 standard
deviations -- which are 1998 and 2022.

Output:
  manuscript/v4/figures/supplement/S_Figure_S3_iobw_scatter.{png,svg}
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
from scipy.stats import spearmanr

LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "manuscript" / "v4" / "figures" / "supplement"
DENGUE = PROJ / "data" / "processed" / "opendengue_vn_annual.csv"
CLIMATE = _REPO_ROOT / "raw_data" / "climate" / "climate_indices.csv"

N_BOOT = 2000
SEED = 20260524


Z_THRESHOLD = 1.5


def threshold_outbreaks(years: np.ndarray, cases: np.ndarray,
                         threshold: float = Z_THRESHOLD) -> list[int]:
    """Outbreak years whose detrended log-case z-score exceeds +threshold."""
    log_cases = np.log(cases)
    x = np.arange(len(log_cases), dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, log_cases, rcond=None)
    detr = log_cases - A @ coef
    z = detr / detr.std(ddof=1)
    return sorted(years[z > threshold].tolist())


def bootstrap_slope_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                      seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        slopes[i] = np.polyfit(x[idx], y[idx], 1)[0]
    point = np.polyfit(x, y, 1)[0]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    od = pd.read_csv(DENGUE)[["year", "cases"]].sort_values("year").reset_index(drop=True)
    ci = pd.read_csv(CLIMATE)
    iobw_mam = (ci[(ci.index_name == "IOBW") & (ci.season == "MAM")]
                [["year", "value"]]
                .rename(columns={"value": "iobw_mam"}))
    df = od.merge(iobw_mam, on="year", how="inner").sort_values("year").reset_index(drop=True)
    df["ln_cases"] = np.log(df["cases"])

    yrs = df["year"].to_numpy()
    cases = df["cases"].to_numpy(dtype=float)
    outbreak_yrs = threshold_outbreaks(yrs, cases, threshold=Z_THRESHOLD)
    print(f"  [S3] outbreak years (detrended log-z > +{Z_THRESHOLD}): "
          f"{outbreak_yrs}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Panel A -- raw scatter
    ax = axes[0]
    rho_a, _ = spearmanr(df.iobw_mam, df.ln_cases)
    slope_a, lo_a, hi_a = bootstrap_slope_ci(df.iobw_mam.to_numpy(),
                                              df.ln_cases.to_numpy())
    # background scatter
    is_out = df.year.isin(outbreak_yrs).to_numpy()
    ax.scatter(df.iobw_mam[~is_out], df.ln_cases[~is_out],
               s=42, color=LANCET_BLUE, edgecolor=LANCET_INK, lw=0.5,
               zorder=3)
    ax.scatter(df.iobw_mam[is_out], df.ln_cases[is_out],
               s=70, color=LANCET_RED, edgecolor=LANCET_INK, lw=0.6,
               zorder=5, label="Outbreak year")
    for _, row in df[is_out].iterrows():
        ax.annotate(str(int(row.year)),
                    xy=(row.iobw_mam, row.ln_cases),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=LANCET_INK)
    # OLS line + bootstrap fan
    xs = np.linspace(df.iobw_mam.min(), df.iobw_mam.max(), 100)
    intercept_a = df.ln_cases.mean() - slope_a * df.iobw_mam.mean()
    ax.plot(xs, slope_a * xs + intercept_a, "-", color=LANCET_RED, lw=1.6,
            zorder=4, label=f"OLS slope {slope_a:+.2f} [{lo_a:+.2f}, {hi_a:+.2f}]")
    ax.set_xlabel("Spring IOBW (degC anomaly, MAM)")
    ax.set_ylabel("ln(reported cases)")
    ax.set_title(f"(A) Raw, rho={rho_a:+.2f}",
                 loc="left", fontsize=9.5, color=LANCET_INK)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # Panel B -- first-differenced scatter
    ax = axes[1]
    d_iobw = np.diff(df.iobw_mam.to_numpy())
    d_ln = np.diff(df.ln_cases.to_numpy())
    d_yrs = df.year.to_numpy()[1:]
    rho_b, _ = spearmanr(d_iobw, d_ln)
    slope_b, lo_b, hi_b = bootstrap_slope_ci(d_iobw, d_ln)
    is_out_b = np.isin(d_yrs, outbreak_yrs)
    ax.axhline(0, color=LANCET_GREY, lw=0.7, zorder=1)
    ax.axvline(0, color=LANCET_GREY, lw=0.7, zorder=1)
    ax.scatter(d_iobw[~is_out_b], d_ln[~is_out_b],
               s=42, color=LANCET_BLUE, edgecolor=LANCET_INK, lw=0.5,
               zorder=3)
    ax.scatter(d_iobw[is_out_b], d_ln[is_out_b],
               s=70, color=LANCET_RED, edgecolor=LANCET_INK, lw=0.6,
               zorder=5)
    for x, y, yr in zip(d_iobw[is_out_b], d_ln[is_out_b], d_yrs[is_out_b]):
        ax.annotate(str(int(yr)), xy=(x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=LANCET_INK)
    xs = np.linspace(d_iobw.min(), d_iobw.max(), 100)
    intercept_b = d_ln.mean() - slope_b * d_iobw.mean()
    ax.plot(xs, slope_b * xs + intercept_b, "-", color=LANCET_RED, lw=1.6,
            zorder=4, label=f"OLS slope {slope_b:+.2f} [{lo_b:+.2f}, {hi_b:+.2f}]")
    ax.set_xlabel("delta spring IOBW")
    ax.set_ylabel("delta ln(cases)")
    ax.set_title(f"(B) First-difference, rho={rho_b:+.2f}",
                 loc="left", fontsize=9.5, color=LANCET_INK)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "S_Figure_S3_iobw_scatter.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "S_Figure_S3_iobw_scatter.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [S3] panel A: rho={rho_a:+.3f}, slope={slope_a:+.3f} "
          f"[{lo_a:+.3f}, {hi_a:+.3f}]")
    print(f"  [S3] panel B: rho={rho_b:+.3f}, slope={slope_b:+.3f} "
          f"[{lo_b:+.3f}, {hi_b:+.3f}]")
    print(f" 