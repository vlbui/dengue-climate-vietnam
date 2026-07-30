"""10 - Figure 3: raw climate-dengue correlations heatmap + lollipop ranking.

Computes Spearman rho between ln(cases) and 16 climate predictors --
8 indices (AMO, DMI, IOBW, MEI, ONI, PDO, SOI, TNA) at 2 seasonal
aggregations (annual mean and spring MAM mean) -- at lag 0 and lag 1.

Panel A: heatmap of rho at lag 0 and lag 1.
Panel B: lollipop of best-of-two-lags rho per predictor; red = BH-FDR
significant at 5%.

No figure-level title (caption added in Word). Outputs:
  figures/static/fig3_raw_correlations.{png,svg}
  tables/raw_correlations.csv
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
LANCET_INK = "#1B1919"
LANCET_GREY = "#ADB6B6"

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "processed"
TABLES = PROJ / "tables"
FIG_STATIC = PROJ / "figures" / "static"
CLIM = Path("/sessions/charming-festive-planck/mnt/gbd/raw_data/climate_data")
# Same path expressed in OS-agnostic form for portability:
CLIM_ALT = PROJ.parents[1] / "raw_data" / "climate_data"

INDICES = ["AMO", "DMI", "IOBW", "MEI", "ONI", "PDO", "SOI", "TNA"]


def spearman_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = np.corrcoef(rx, ry)[0, 1]
    # 2-sided p via t-statistic approximation
    if abs(rho) >= 1 or n < 4:
        return float(rho), 1.0
    t = rho * np.sqrt((n - 2) / (1 - rho * rho))
    # Use approximation: p ~ 2 * (1 - Phi(|t| * (1 - 1/(4n)))) is not exact
    # Use scipy-free approximation: survival of student-t via normal approx for n>=10
    from math import erf, sqrt
    z = abs(t)
    # Normal approx (rough but adequate for ranking)
    p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return float(rho), float(p)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(adj, 0, 1)
    return out


def load_index(name: str) -> pd.DataFrame:
    path_options = [
        CLIM / f"{name.lower()}.csv",
        CLIM_ALT / f"{name.lower()}.csv",
    ]
    for p in path_options:
        if p.exists():
            df = pd.read_csv(p, comment="#")
            break
    else:
        raise FileNotFoundError(f"no climate file found for {name}")
    # Normalise column names
    cols = {c.strip().lower(): c for c in df.columns}
    if "year" in cols and "month" in cols and "value" in cols:
        df = df.rename(columns={cols["year"]: "year",
                                cols["month"]: "month",
                                cols["value"]: "value"})
        df = df[["year", "month", "value"]].dropna()
    elif "date" in cols:
        df = df.rename(columns={cols["date"]: "date"})
        d = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = d.dt.year
        df["month"] = d.dt.month
        # value = first non-date numeric column
        val_col = [c for c in df.columns if c not in ("date", "year", "month")][0]
        df["value"] = pd.to_numeric(df[val_col], errors="coerce")
        df = df[["year", "month", "value"]].dropna()
    else:
        # Two-column CSV: assume first = date-like, second = value
        c0, c1 = df.columns[:2]
        d = pd.to_datetime(df[c0], errors="coerce")
        df["year"] = d.dt.year
        df["month"] = d.dt.month
        df["value"] = pd.to_numeric(df[c1], errors="coerce")
        df = df[["year", "month", "value"]].dropna()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


def aggregate_seasonal(df: pd.DataFrame, season: str) -> pd.Series:
    """Return Series indexed by year for given seasonal aggregation."""
    if season == "annual":
        s = df.groupby("year")["value"].mean()
        return s
    elif season == "MAM":
        sel = df[df["month"].isin([3, 4, 5])]
        s = sel.groupby("year")["value"].mean()
        return s
    raise ValueError(season)


def main():
    FIG_STATIC.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    od = pd.read_csv(DATA / "opendengue_vn_annual.csv")
    od = od.sort_values("year").reset_index(drop=True)
    od["log_cases"] = np.log(od["cases"])
    od_y = od.set_index("year")["log_cases"]

    rows = []
    for idx in INDICES:
        try:
            raw = load_index(idx)
        except FileNotFoundError as e:
            print(f"  WARN: skipping {idx}: {e}")
            continue
        for season in ("annual", "MAM"):
            s = aggregate_seasonal(raw, season)
            for lag in (0, 1):
                # align: predictor at year t-lag vs ln(cases) at year t
                shifted = s.copy()
                shifted.index = shifted.index + lag
                merged = pd.concat([od_y, shifted.rename("x")], axis=1).dropna()
                if len(merged) < 8:
                    continue
                rho, p = spearman_rho(merged["x"].to_numpy(),
                                      merged["log_cases"].to_numpy())
                rows.append(dict(predictor=idx, season=season, lag=lag,
                                 n=len(merged), rho=rho, p=p))
    df = pd.DataFrame(rows)

    # Best-of-two-lags rho per (predictor, season)
    df["abs_rho"] = df["rho"].abs()
    best = (df.sort_values("abs_rho", ascending=False)
              .groupby(["predictor", "season"], as_index=False)
              .first()
              .drop(columns="abs_rho"))
    # BH-FDR on the best-of-two-lags p-values
    best["p_BH"] = bh_fdr(best["p"].to_numpy())
    best["sig5"] = best["p_BH"] < 0.05
    best.to_csv(TABLES / "raw_correlations.csv", index=False)
    print(f"  [10] N predictors x seasons = {len(best)};  significant: {best.sig5.sum()}")

    # ---- Panel A: heatmap (16 rows x 2 lag cols) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    rows_order = []
    for idx in INDICES:
        for season in ("annual", "MAM"):
            label_season = "annual" if season == "annual" else "spring"
            rows_order.append((idx, season, f"{idx} ({label_season})"))
    mat = np.full((len(rows_order), 2), np.nan)
    for i, (idx, season, _label) in enumerate(rows_order):
        for j, lag in enumerate((0, 1)):
            sel = df[(df.predictor == idx) & (df.season == season) & (df.lag == lag)]
            if not sel.empty:
                mat[i, j] = sel.iloc[0].rho
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_xlabel("Lag (years)")
    ax.set_yticks(np.arange(len(rows_order)))
    ax.set_yticklabels([r[2] for r in rows_order], fontsize=8)
    ax.set_title("(A) Lagged rho", loc="left", fontsize=10, color=LANCET_INK)
    for i in range(len(rows_order)):
        for j in range(2):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > 0.45 else LANCET_INK)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman rho")

    # ---- Panel B: lollipop ranking ----
    ax = axes[1]
    rank = best.sort_values("rho", ascending=False).reset_index(drop=True)
    y = np.arange(len(rank))[::-1]
    colors = [LANCET_RED if s else LANCET_GREY for s in rank.sig5]
    ax.hlines(y, 0, rank.rho, colors=colors, lw=2)
    ax.plot(rank.rho, y, "o", color=LANCET_INK, ms=4)
    ax.axvline(0, color=LANCET_INK, lw=0.6)
    labels = [f"{r.predictor} ({'annual' if r.season == 'annual' else 'spring'}, L{r.lag})"
              for r in rank.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho (best-of-two-lags; red = FDR 5%)")
    ax.set_title("(B) Predictor ranking", loc="left", fontsize=10, color=LANCET_INK)
    ax.grid(True, ls="--", alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIG_STATIC / "fig3_raw_correlations.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_STATIC / "fig3_raw_correlations.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [10] figure: figures/static/fig3_raw_correlations.png + .svg")


def verify():
    p = FIG_STATIC / "fig3_raw_correlations.png"
    assert p.exists(), f"missing {p}"
    sz = p.stat().st_size
    print(f"  [verify] {p.name}: {sz/1024:.0f} KB")
    assert sz > 50_000, "figure file suspiciously small"


if __name__ == "__main__":
    main()
    verify()
