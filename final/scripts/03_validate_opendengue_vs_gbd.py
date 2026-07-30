"""03 — Validate OpenDengue surveillance against GBD modelled incidence.

Compares the merged OpenDengue annual series (manuscript v3 Tier 1–7) with
IHME GBD 2023 modelled all-ages dengue incidence for Vietnam, 1990–2023.

Surveillance counts and modelled estimates differ structurally — IHME
applies under-reporting and severity adjustments — so we expect GBD > raw
counts. The point of this script is to quantify the gap and check that it
is stable (or trends in a known direction) rather than fluctuating
randomly across the seven OpenDengue source tiers.

Outputs:
  tables/validation_opendengue_vs_gbd.csv
  tables/validation_summary_by_source.csv
  figures/static/fig_validation.{png,svg}
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

# Lancet palette (mirrors shared.palette without importing scipy chain).
LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_GREEN = "#42B540"
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "processed"
TABLES = PROJ / "tables"
FIG_STATIC = PROJ / "figures" / "static"


def spearman_rho(x, y):
    """Spearman rho via rank-Pearson; avoids hard scipy dependency."""
    x = pd.Series(x).rank()
    y = pd.Series(y).rank()
    return float(x.corr(y))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG_STATIC.mkdir(parents=True, exist_ok=True)

    od = pd.read_csv(DATA / "opendengue_vn_annual.csv")
    gbd = pd.read_csv(DATA / "gbd_dengue_vn.csv")

    inc = (gbd[(gbd.measure_name == "Incidence")
               & (gbd.metric_name == "Number")
               & (gbd.age_name == "All ages")]
           [["year", "val", "lower", "upper"]]
           .rename(columns={"val": "gbd_inc",
                            "lower": "gbd_inc_lo",
                            "upper": "gbd_inc_hi"}))

    merged = od.merge(inc, on="year", how="inner")
    merged["ratio_gbd_to_od"] = merged["gbd_inc"] / merged["cases"]
    merged["log_ratio"] = np.log10(merged["ratio_gbd_to_od"])
    merged["mean_log"] = (np.log10(merged["cases"])
                          + np.log10(merged["gbd_inc"])) / 2
    merged["diff_log"] = (np.log10(merged["gbd_inc"])
                          - np.log10(merged["cases"]))

    merged.to_csv(TABLES / "validation_opendengue_vs_gbd.csv", index=False)
    print(f"  [03] paired rows: {len(merged)} years "
          f"({merged.year.min()}-{merged.year.max()})")

    # Summary by source label
    rows = []
    for src, sub in merged.groupby("source_label"):
        rho = spearman_rho(sub["cases"], sub["gbd_inc"]) if len(sub) >= 3 else float("nan")
        rows.append(dict(
            source_label=src,
            n_years=len(sub),
            year_min=int(sub.year.min()),
            year_max=int(sub.year.max()),
            mean_ratio=sub["ratio_gbd_to_od"].mean(),
            median_ratio=sub["ratio_gbd_to_od"].median(),
            spearman=rho,
        ))
    summ = pd.DataFrame(rows).sort_values("year_min").reset_index(drop=True)
    summ.to_csv(TABLES / "validation_summary_by_source.csv", index=False)

    print("  [03] GBD/OD ratio by source label:")
    for _, r in summ.iterrows():
        rho_str = f"{r['spearman']:+.2f}" if not pd.isna(r['spearman']) else "  n/a"
        print(f"           {r['source_label']:<22} "
              f"n={int(r['n_years']):>2}  "
              f"{int(r['year_min'])}-{int(r['year_max'])}  "
              f"median={r['median_ratio']:>5.2f}x  "
              f"rho={rho_str}")

    overall_rho = spearman_rho(merged["cases"], merged["gbd_inc"])
    print(f"  [03] overall Spearman OpenDengue<->GBD: {overall_rho:+.3f}")

    _build_figure(merged, summ, overall_rho)


def _build_figure(merged: pd.DataFrame, summ: pd.DataFrame, overall_rho: float):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    src_order = list(summ.sort_values("year_min")["source_label"])
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(src_order)}

    # Panel A: paired time series (log y)
    ax = axes[0]
    ax.plot(merged.year, merged.gbd_inc, color=LANCET_BLUE, lw=1.6,
            marker="o", ms=4, label="GBD modelled incidence")
    ax.fill_between(merged.year, merged.gbd_inc_lo, merged.gbd_inc_hi,
                    color=LANCET_BLUE, alpha=0.15, lw=0)
    for src, sub in merged.groupby("source_label", sort=False):
        ax.scatter(sub.year, sub.cases, s=35, color=colors[src],
                   edgecolor=LANCET_INK, lw=0.5, zorder=3, label=f"OD: {src}")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cases (log)")
    ax.set_title("(A) Paired annual series, 1990-2023")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(True, ls="--", alpha=0.3)

    # Panel B: scatter OpenDengue vs GBD (log-log)
    ax = axes[1]
    for src, sub in merged.groupby("source_label", sort=False):
        ax.scatter(sub.cases, sub.gbd_inc, s=45, color=colors[src],
                   edgecolor=LANCET_INK, lw=0.5, label=src)
    lim = [merged[["cases", "gbd_inc"]].min().min() * 0.7,
           merged[["cases", "gbd_inc"]].max().max() * 1.3]
    ax.plot(lim, lim, ls="--", color=LANCET_INK, lw=0.8, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("OpenDengue cases (Number)")
    ax.set_ylabel("GBD modelled incidence (Number)")
    ax.set_title(f"(B) OD vs GBD  (Spearman rho = {overall_rho:+.2f})")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(True, ls="--", alpha=0.3, which="both")

    # Panel C: Bland-Altman on log10 (differences vs means)
    ax = axes[2]
    diff = merged["diff_log"].values
    md, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    for src, sub in merged.groupby("source_label", sort=False):
        ax.scatter(sub["mean_log"], sub["diff_log"],
                   s=45, color=colors[src],
                   edgecolor=LANCET_INK, lw=0.5, label=src)
    ax.axhline(md, color=LANCET_RED, lw=1.2,
               label=f"mean diff = {md:+.2f} (~{10**md:.1f}x)")
    ax.axhline(md + 1.96 * sd, color=LANCET_RED, lw=0.8, ls="--",
               label=f"+/-1.96 SD ({sd:.2f})")
    ax.axhline(md - 1.96 * sd, color=LANCET_RED, lw=0.8, ls="--")
    ax.set_xlabel("mean(log10 OD, log10 GBD)")
    ax.set_ylabel("log10 GBD - log10 OD")
    ax.set_title("(C) Bland-Altman (log10)")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()

    fig.savefig(FIG_STATIC / "fig_validation.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_STATIC / "fig_validation.svg", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  [03] figure: figures/static/fig_validation.png + .svg")


def verify():
    """Sanity check: GBD should systematically exceed OpenDengue counts."""
    summ = pd.read_csv(TABLES / "validation_summary_by_source.csv")
    overall_median = summ["median_ratio"].median()
    print(f"  [verify] median GBD/OD ratio across all source labels: "
          f"{overall_median:.1f}x")
    assert overall_median > 1.0, \
        "GBD modelled incidence should exceed surveillance counts"
    print("  [verify] GBD > OD invariant holds (under-reporting "
          "consistent with literature)")


if __name__ == "__main__":
    main()
    verify()
    main()
    verify()
