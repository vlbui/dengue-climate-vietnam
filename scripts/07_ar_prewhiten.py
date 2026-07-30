"""07 - AR-prewhitening robustness check on the top-two climate findings.

The discussion paragraph in manuscript v3 flags AR-prewhitening as a missing
robustness check, because the serial-correlation structure of dengue can produce
lag-1 artefacts. This script implements that check on the two predictor x lag
combinations that headline the climate section: annual AMO at lag 0 and spring
TNA at lag 1.

Procedure (per the manuscript's promised limitation check):
  1. Fit AR(1) to ln(cases) over 1990-2023 (statsmodels).
  2. Form prewhitened residuals  e_t = ln_cases_t - rho_1 * ln_cases_{t-1}.
     Whatever co-moves with the climate index in this residual cannot be
     attributed to the dengue series's own first-order persistence.
  3. Re-correlate e_t with annual AMO at lag 0 and spring (MAM) TNA at lag 1
     using Spearman rho with a percentile bootstrap 95% CI (n_boot=1000).

Outputs:
  tables/ar_prewhitened_top2.csv
        predictor, season, lag, n, rho, ci_lo, ci_hi

Headline interpretation: a CI that excludes 0 with the same sign as the raw
finding strengthens the claim; a CI that brackets 0 weakens it. Given that
the SDI-residualised strip already pushes both predictors to non-significance,
this script is the third independent robustness lens, not the last word.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

PROJ = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2].parent  # d:/gbd
DATA = PROJ / "data" / "processed"
TABLES = PROJ / "tables"
CLIMATE_FILE = ROOT / "raw_data" / "climate" / "climate_indices.csv"

N_BOOT = 1000
SEED = 42


def spearman_rho(x, y) -> float:
    """Spearman rho via rank-Pearson; matches the convention in script 05."""
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rhos[i] = spearman_rho(x[idx], y[idx])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    # ── Load dengue ln(cases) over the SDI-residual window ───────────────
    df = pd.read_csv(DATA / "dengue_vn_sdi_residuals.csv").sort_values("year").reset_index(drop=True)
    ln = df["ln_cases"].to_numpy()
    yrs = df["year"].to_numpy()

    # ── Fit AR(1) and prewhiten ──────────────────────────────────────────
    model = AutoReg(ln, lags=1, trend="c").fit()
    # params order with trend='c': [intercept, L1.coef]
    rho_1 = float(model.params[1])
    print(f"  [07] AR(1) coefficient on ln(cases) 1990-2023: rho_1 = {rho_1:+.4f}")
    e = ln[1:] - rho_1 * ln[:-1]
    e_years = yrs[1:]
    print(f"  [07] prewhitened residual: n={len(e)} ({e_years[0]}-{e_years[-1]})")

    e_df = pd.DataFrame({"year": e_years, "e": e})

    # ── Load climate predictors ──────────────────────────────────────────
    clim = pd.read_csv(CLIMATE_FILE)

    rows = []

    # 1) annual AMO, lag 0 -- pair e_t with AMO_t
    amo = clim[(clim["index_name"] == "AMO") & (clim["season"] == "annual")][["year", "value"]]
    j = e_df.merge(amo, on="year", how="inner").dropna()
    rho = spearman_rho(j["e"], j["value"])
    lo, hi = bootstrap_ci(j["e"].to_numpy(), j["value"].to_numpy())
    rows.append(dict(predictor="AMO", season="annual", lag=0, n=len(j),
                     rho=rho, ci_lo=lo, ci_hi=hi))
    print(f"  [07] AMO annual lag 0:  rho={rho:+.3f}  CI95=({lo:+.3f}, {hi:+.3f})  n={len(j)}")

    # 2) spring (MAM) TNA, lag 1 -- pair e_t with TNA_{t-1}
    tna = clim[(clim["index_name"] == "TNA") & (clim["season"] == "MAM")][["year", "value"]].copy()
    tna["year"] = tna["year"] + 1  # shift forward so e_t merges with TNA_{t-1}
    j = e_df.merge(tna, on="year", how="inner").dropna()
    rho = spearman_rho(j["e"], j["value"])
    lo, hi = bootstrap_ci(j["e"].to_numpy(), j["value"].to_numpy())
    rows.append(dict(predictor="TNA", season="MAM", lag=1, n=len(j),
                     rho=rho, ci_lo=lo, ci_hi=hi))
    print(f"  [07] TNA MAM    lag 1:  rho={rho:+.3f}  CI95=({lo:+.3f}, {hi:+.3f})  n={len(j)}")

    out = (pd.DataFrame(rows)
           [["predictor", "season", "lag", "n", "rho", "ci_lo", "ci_hi"]])
    out.to_csv(TABLES / "ar_prewhitened_top2.csv", index=False)
    print(f"  [07] table: tables/ar_prewhitened_top2.csv")

    # ── One-line headline (per spec) ─────────────────────────────────────
    amo_r = out[(out.predictor == "AMO")].iloc[0]
    if amo_r.rho > 0 and amo_r.ci_lo > 0:
        print(f"  [07] -> AMO rho stays positive with 95% CI excluding 0 -- finding strengthened.")
    elif amo_r.ci_lo <= 0 <= amo_r.ci_hi:
        print(f"  [07] -> AMO 95% CI brackets 0 (rho={amo_r.rho:+.3f}, CI=({amo_r.ci_lo:+.3f}, {amo_r.ci_hi:+.3f})) "
              f"-- AR-prewhitened association is not distinguishable from null.")
    else:
        print(f"  [07] -> AMO CI excludes 0 with sign {'positive' if amo_r.rho > 0 else 'negative'}; "
              f"flag if direction reversed.")


def verify():
    out = pd.read_csv(TABLES / "ar_prewhitened_top2.csv")
    assert list(out.columns) == ["predictor", "season", "lag", "n", "rho", "ci_lo", "ci_hi"], out.columns
    assert len(out) == 2, len(out)
    assert set(zip(out["predictor"], out["lag"])) == {("AMO", 0), ("TNA", 1)}
    assert out[["rho", "ci_lo", "ci_hi"]].notna().all().all()
    # Sanity: ci_lo <= rho <= ci_hi for every row
    for _, r in out.iterrows():
        assert r["ci_lo"] <= r["rho"] <= r["ci_hi"], r.to_dict()
    print(f"  [verify] 2 rows; CI ordering OK; predictors = {out['predictor'].tolist()}")


if __name__ == "__main__":
    main()
    verify()
