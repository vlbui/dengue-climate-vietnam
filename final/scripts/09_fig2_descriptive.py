"""09 - Figure 2: two-panel descriptive summary for Vietnam dengue.

Post-review rewrite (2026-05-24): the previous three-panel layout had
Panel A (log cases with outbreak markers) and Panel B (detrended
z-scores) showing the same series before and after trend removal. They
are now merged into one panel. The log-case series is drawn with its
fitted linear trend and a shaded band at plus/minus 1.5 standard
deviations of the detrended series. Years whose reported count falls
above the band are outbreak years (red); years below it are trough
years (teal). Because the z = plus/minus 1.5 threshold maps exactly
onto a plus/minus 1.5 SD band around the trend in log-case space, this
single panel carries the full information of the old Panels A and B
without needing a second y-axis.

Panels (no figure-level title; panel labels A/B kept as small subtitle
text so the caption can be added in Word).
  (A) Reported cases on a log scale, with the fitted linear trend, a
      plus/minus 1.5 SD band, outbreak and trough years, and policy
      events as green dotted vertical lines.
  (B) Autocorrelation of detrended log cases (lags 1 to 12) with the
      approximate 95 percent white-noise significance bounds shaded.

Plain-English convention: axis labels, panel titles, and in-figure text
avoid mathematical notation. The word "log" refers to the natural
logarithm.

Outputs:
  figures/static/fig2_descriptive.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_GREEN = "#42B540"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"
LANCET_TEAL = "#0099B4"

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "processed"
FIG_STATIC = PROJ / "figures" / "static"

# Outbreak / trough years -- statistically defined: detrended log
# z-score beyond +/-1.5. Only 1998 and 2022 exceed +1.5; 1999, 2000
# and 2014 fall below -1.5. These are exactly the years that fall
# outside the +/-1.5 SD band drawn in panel A.
OUTBREAK_YEARS = [1998, 2022]
TROUGH_YEARS = [1999, 2000, 2014]

# Surveillance / policy milestones (year, label) -- verified events that
# could plausibly cause a step change in reported case counts.
POLICY_EVENTS = [
    (2009, "WHO dengue case-classification revision"),
    (2016, "National electronic surveillance (eCDS) launch"),
    (2020, "COVID-19 pandemic disruption"),
]


def linear_fit(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (trend, residual) from an ordinary-least-squares line fit."""
    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    trend = A @ coef
    return trend, y - trend


def acf(y: np.ndarray, nlags: int = 12) -> np.ndarray:
    y = y - y.mean()
    n = len(y)
    var = (y * y).sum() / n
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = (y[: n - k] * y[k:]).sum() / (n * var)
    return out


def main():
    FIG_STATIC.mkdir(parents=True, exist_ok=True)

    od = pd.read_csv(DATA / "opendengue_vn_annual.csv")
    od = od.sort_values("year").reset_index(drop=True)
    yrs = od["year"].to_numpy(dtype=int)
    cases = od["cases"].to_numpy(dtype=float)
    log_cases = np.log(cases)

    trend, detr = linear_fit(log_cases)
    sigma = detr.std(ddof=1)
    mu = detr.mean()
    # +/-1.5 SD band around the trend, in log-case space. A year above
    # band_hi has detrended z > +1.5 (outbreak); below band_lo, z < -1.5
    # (trough).
    band_hi = trend + mu + 1.5 * sigma
    band_lo = trend + mu - 1.5 * sigma

    # 1 row x 2 columns, equal-width panels.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # (A) Merged panel: log-scale cases with fitted trend, +/-1.5 SD band,
    #     outbreak/trough markers and policy-event vertical lines. This
    #     replaces the old Panel A (log cases) and Panel B (detrended
    #     z-scores) of the three-panel layout.
    ax = axes[0]
    is_out = np.isin(yrs, OUTBREAK_YEARS)
    is_tro = np.isin(yrs, TROUGH_YEARS)
    is_norm = ~is_out & ~is_tro
    # +/-1.5 SD band
    ax.fill_between(yrs, band_lo, band_hi, color=LANCET_GREY, alpha=0.30,
                    linewidth=0, zorder=1,
                    label="±1.5 SD band (detrended)")
    # fitted linear trend
    ax.plot(yrs, trend, "--", color=LANCET_INK, lw=1.2, alpha=0.85,
            zorder=2, label="Linear trend")
    # baseline case series
    ax.plot(yrs, log_cases, "-", color=LANCET_BLUE, lw=1.4, zorder=3)
    # ordinary-year markers
    ax.plot(yrs[is_norm], log_cases[is_norm], "o", color=LANCET_BLUE,
            ms=4, zorder=4)
    # outbreak markers (above the band)
    ax.plot(yrs[is_out], log_cases[is_out], "o", color=LANCET_RED, ms=7,
            markeredgecolor=LANCET_INK, markeredgewidth=0.6, zorder=5,
            label="Outbreak year (above band)")
    # trough markers (below the band)
    ax.plot(yrs[is_tro], log_cases[is_tro], "o", color=LANCET_TEAL, ms=7,
            markeredgecolor=LANCET_INK, markeredgewidth=0.6, zorder=5,
            label="Trough year (below band)")
    # policy events as dotted green vertical lines with year labels
    ylim_lo = min(log_cases.min(), band_lo.min()) - 0.5
    ylim_hi = max(log_cases.max(), band_hi.max()) + 0.45
    ax.set_ylim(ylim_lo, ylim_hi)
    for yr, _lbl in POLICY_EVENTS:
        ax.axvline(yr, color=LANCET_GREEN, ls=":", lw=1.2, alpha=0.9,
                   zorder=2)
        ax.text(yr, ylim_hi - 0.05, f" {yr}", fontsize=7,
                color=LANCET_GREEN, rotation=90, va="top", ha="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Reported cases (log scale)")
    ax.set_title("(A) Annual reported cases with trend and ±1.5 SD band",
                 loc="left", fontsize=9.5, color=LANCET_INK)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(True, ls="--", alpha=0.3)

    # (B) Autocorrelation of detrended log cases
    ax = axes[1]
    nlags = 12
    a = acf(detr, nlags=nlags)
    ci = 1.96 / np.sqrt(len(detr))
    lags = np.arange(nlags + 1)
    ax.axhline(0, color=LANCET_INK, lw=0.6)
    ax.axhspan(-ci, ci, color=LANCET_GREY, alpha=0.25, linewidth=0,
               label="Approximate 95% significance bounds (white-noise null)")
    ax.vlines(lags, 0, a, colors=LANCET_BLUE, lw=2)
    ax.plot(lags, a, "o", color=LANCET_BLUE, ms=4)
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(lags)
    ax.set_title("(B) Autocorrelation of detrended log cases",
                 loc="left", fontsize=9.5, color=LANCET_INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, ls="--", alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_STATIC / "fig2_descriptive.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_STATIC / "fig2_descriptive.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [09] figure: figures/static/fig2_descriptive.png + .svg "
          f"(two-panel layout)")
    print(f"  [09] N = {len(yrs)} years, range {yrs[0]}-{yrs[-1]}; "
          f"outbreak {OUTBREAK_YEARS}, trough {TROUGH_YEARS}")
    print(f"  [09] detrended SD = {sigma:.3f}; band = trend +/- "
          f"{1.5 * sigma:.3f} (log units)")


def verify():
    p = FIG_STATIC / "fig2_descriptive.png"
    assert p.exists(), f"missing {p}"
    sz = p.stat().st_size
    print(f"  [verify] {p.name}: {sz/1024:.0f} KB")
    assert sz > 50_000, "figure file suspiciously small"


if __name__ == "__main__":
    main()
    verify()
