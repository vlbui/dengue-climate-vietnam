"""16 - AR(1)-prewhitening + bootstrap robustness check on ALL first-differencing survivors.

Motivation
----------
Script 07 (ar_prewhiten) applied the AR(1) persistence check to only two,
hand-picked predictors (annual AMO at lag 0 and spring TNA at lag 1). That
asymmetry meant the leading raw signal, IOBW, was never put through the same
test even though it survives first-differencing more strongly than TNA.

This script removes the hand-picking: it reads the first-differencing survivors
straight from the trend-correction table (rho_firstdiff_pBH < 0.05) and runs the
identical AR(1) + bootstrap procedure on every one of them, so all candidates are
judged by the same rule.

Procedure (identical to script 07)
----------------------------------
  1. Fit AR(1) to ln(cases) over the residual window (AutoReg, trend='c'; a numpy
     OLS fallback is used if statsmodels is unavailable - the estimate is the same).
  2. Prewhiten:  e_t = ln_cases_t - rho_1 * ln_cases_{t-1}.
  3. For each survivor (index, season, lag), pair e_t with the index value at
     year (t - lag) and compute Spearman rho with a percentile bootstrap 95% CI.

Verdict: CI excludes 0 -> survives AR(1); CI brackets 0 -> does not.

Output: tables/ar_prewhitened_all_survivors.csv
        index_name, season, lag, n, rho_firstdiff, rho_ar1, ci_lo, ci_hi, ar1_survives
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Paths: resolve relative to this script (expects to live in <final>/scripts) ---
PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data_inputs"
TABLES = PROJ / "tables"
RESID_FILE = DATA / "dengue_vn_sdi_residuals.csv"
CLIMATE_FILE = DATA / "climate_indices.csv"
STRIP_FILE = TABLES / "TableS3_trend_correction_comparison.csv"

N_BOOT = 1000
SEED = 42


def ar1_coef(ln: np.ndarray) -> float:
    """AR(1) coefficient (L1) of ln with a constant. Uses statsmodels if present,
    otherwise an equivalent closed-form OLS slope."""
    try:
        from statsmodels.tsa.ar_model import AutoReg
        return float(AutoReg(ln, lags=1, trend="c").fit().params[1])
    except Exception:
        x, y = ln[:-1], ln[1:]
        return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def spearman_rho(x, y) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def bootstrap_ci(x, y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rhos[i] = spearman_rho(x[idx], y[idx])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    # 1) ln(cases) and AR(1) prewhitening
    df = pd.read_csv(RESID_FILE).sort_values("year").reset_index(drop=True)
    ln, yrs = df["ln_cases"].to_numpy(), df["year"].to_numpy()
    rho_1 = ar1_coef(ln)
    e = ln[1:] - rho_1 * ln[:-1]
    e_df = pd.DataFrame({"year": yrs[1:], "e": e})
    print(f"  [16] AR(1) coefficient rho_1 = {rho_1:+.4f}  "
          f"(prewhitened residual: n={len(e)}, {yrs[1]}-{yrs[-1]})")

    # 2) first-differencing survivors (same rule as the supplement: BH p < 0.05)
    strip = pd.read_csv(STRIP_FILE)
    surv = (strip[strip["rho_firstdiff_pBH"] < 0.05]
            .sort_values("rho_firstdiff_pBH")
            [["index_name", "season", "lag", "rho_firstdiff"]]
            .reset_index(drop=True))
    print(f"  [16] {len(surv)} first-differencing survivors read from {STRIP_FILE.name}")

    # 3) AR(1) + bootstrap for every survivor
    clim = pd.read_csv(CLIMATE_FILE)
    out = []
    for _, s in surv.iterrows():
        idx, season, lag = s["index_name"], s["season"], int(s["lag"])
        c = clim[(clim["index_name"] == idx) & (clim["season"] == season)][["year", "value"]].copy()
        c["year"] = c["year"] + lag                       # pair e_t with index_{t-lag}
        j = e_df.merge(c, on="year", how="inner").dropna()
        x, y = j["e"].to_numpy(), j["value"].to_numpy()
        rho = spearman_rho(x, y)
        lo, hi = bootstrap_ci(x, y)
        survives = bool(lo > 0 or hi < 0)
        out.append(dict(index_name=idx, season=season, lag=lag, n=len(j),
                        rho_firstdiff=round(float(s["rho_firstdiff"]), 3),
                        rho_ar1=round(rho, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                        ar1_survives=survives))

    res = pd.DataFrame(out)
    res.to_csv(TABLES / "ar_prewhitened_all_survivors.csv", index=False)

    # 4) report
    print("\n  predictor           first-diff   AR(1) rho      95% CI           AR(1)?")
    print("  " + "-" * 70)
    for r in out:
        name = f"{r['index_name']} {r['season']} L{r['lag']}"
        ci = f"({r['ci_lo']:+.2f}, {r['ci_hi']:+.2f})"
        verdict = "SURVIVES" if r["ar1_survives"] else "  ns"
        print(f"  {name:18}  {r['rho_firstdiff']:+.3f}      {r['rho_ar1']:+.3f}   {ci:>16}   {verdict}")
    n_pass = sum(r["ar1_survives"] for r in out)
    print(f"\n  [16] {n_pass}/{len(out)} first-differencing survivors also survive AR(1) prewhitening.")
    print(f"  [16] table: tables/ar_prewhitened_all_survivors.csv")


if __name__ == "__main__":
    main()
