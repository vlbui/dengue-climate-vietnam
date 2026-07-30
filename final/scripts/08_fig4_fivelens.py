"""08 - Figure 4 (NEW): five-lens trend-correction comparison bar chart.

Visualises Spearman rho for the seven raw BH-FDR-significant predictors
under five trend-correction lenses (raw, linear-detrended, first-differenced,
SDI-residualised, AR(1)-prewhitened where computed). The annual AMO at
lag 0 row illustrates the diagnostic disagreement: AMO survives the
persistence-stripping lenses but not the trend-stripping lenses.

Outputs:
  figures/static/fig4_fivelens.{png,svg}
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
LANCET_PURPLE = "#925E9F"
LANCET_TEAL = "#0099B4"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

PROJ = Path(__file__).resolve().parents[1]
TABLES = PROJ / "tables"
FIG_STATIC = PROJ / "figures" / "static"

# Top 7 raw BH-FDR-significant predictors at lag 0 (and TNA at lag 1)
TOP7 = [
    ("AMO", "annual", 0),
    ("TNA", "annual", 0),
    ("IOBW", "annual", 0),
    ("AMO", "MAM", 0),
    ("IOBW", "MAM", 0),
    ("DMI", "MAM", 0),
    ("DMI", "annual", 0),
]

LENS_ORDER = ["raw", "linear-detrended", "first-differenced",
              "SDI-residualised", "AR(1)-prewhitened"]
LENS_COLORS = {
    "raw": LANCET_GREY,
    "linear-detrended": LANCET_TEAL,
    "first-differenced": LANCET_GREEN,
    "SDI-residualised": LANCET_BLUE,
    "AR(1)-prewhitened": LANCET_RED,
}


def main():
    FIG_STATIC.mkdir(parents=True, exist_ok=True)

    sdi = pd.read_csv(TABLES / "sdi_strip_comparison.csv")
    ar = pd.read_csv(TABLES / "ar_prewhitened_top2.csv")

    # Build matrix: rows = predictors (TOP7), cols = lenses
    rows = []
    for idx, season, lag in TOP7:
        sdi_row = sdi[(sdi.index_name == idx) & (sdi.season == season)
                      & (sdi.lag == lag)]
        if sdi_row.empty:
            continue
        r = sdi_row.iloc[0]
        ar_row = ar[(ar.predictor == idx) & (ar.season == season)
                    & (ar.lag == lag)]
        ar_rho = float(ar_row.iloc[0].rho) if not ar_row.empty else np.nan

        label = f"{idx} {season} L{lag}"
        rows.append(dict(
            label=label,
            raw=float(r.rho_raw),
            **{"linear-detrended": float(r.rho_lineardetrend)},
            **{"first-differenced": float(r.rho_firstdiff)},
            **{"SDI-residualised": float(r.rho_sdi_residual)},
            **{"AR(1)-prewhitened": ar_rho},
            raw_pBH=float(r.rho_raw_pBH),
            sdi_pBH=float(r.rho_sdi_residual_pBH),
            firstdiff_pBH=float(r.rho_firstdiff_pBH),
            lin_pBH=float(r.rho_lineardetrend_pBH),
        ))
    df = pd.DataFrame(rows)
    print(f"  [08] {len(df)} predictors x 5 lenses")

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_pred = len(df)
    n_lens = len(LENS_ORDER)
    bar_width = 0.16
    x = np.arange(n_pred)

    for i, lens in enumerate(LENS_ORDER):
        offset = (i - (n_lens - 1) / 2) * bar_width
        vals = df[lens].values
        bars = ax.bar(x + offset, vals, bar_width,
                      color=LENS_COLORS[lens], edgecolor=LANCET_INK,
                      linewidth=0.4, label=lens)
        # Mark BH-FDR significance at 5%
        for j, v in enumerate(vals):
            pBH_col = {"raw": "raw_pBH", "linear-detrended": "lin_pBH",
                       "first-differenced": "firstdiff_pBH",
                       "SDI-residualised": "sdi_pBH"}.get(lens)
            if pBH_col and not np.isnan(v):
                pBH = df.iloc[j][pBH_col]
                if pBH < 0.05:
                    sign_y = v + 0.02 if v > 0 else v - 0.05
                    ax.text(x[j] + offset, sign_y, "*",
                            ha="center", va="bottom" if v > 0 else "top",
                            fontsize=11, color=LANCET_INK, weight="bold")
            elif lens == "AR(1)-prewhitened" and not np.isnan(v):
                # AR-prewhitened uses CI from ar_prewhitened_top2.csv
                # AMO annual lag 0 has CI excluding 0 (significant)
                if "AMO annual" in df.iloc[j].label:
                    ax.text(x[j] + offset, v + 0.02, "*",
                            ha="center", va="bottom", fontsize=11,
                            color=LANCET_INK, weight="bold")

    ax.axhline(0, color=LANCET_INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Spearman rho with ln(cases)")
    ax.set_ylim(-0.2, 0.65)
    ax.legend(fontsize=8, loc="lower right", ncol=2, frameon=False)
    ax.grid(True, ls="--", alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_STATIC / "fig4_fivelens.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_STATIC / "fig4_fivelens.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [08] figure: figures/static/fig4_fivelens.png + .svg")


def verify():
    p = FIG_STATIC / "fig4_fivelens.png"
    assert p.exists(), f"missing {p}"
    sz = p.stat().st_size
    print(f"  [verify] {p.name}: {sz/1024:.0f} KB")
    assert sz > 50_000, "figure file suspiciously small"


if __name__ == "__main__":
    main()
    verify()
