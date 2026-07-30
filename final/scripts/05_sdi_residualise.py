"""05 - SDI residualisation: a domain-informed alternative to linear detrending.

Manuscript v3 currently uses linear detrending and first-differencing as
its two trend-correction strips. Both are atheoretic: a critic can ask
"what does the trend mean?". This script adds a third strip that
regresses ln(cases) on Vietnam SDI (Socio-demographic Index, IHME) and
takes the residuals. Whatever survives in the residual series cannot be
attributed to socioeconomic development (which captures urbanisation,
healthcare growth, surveillance growth, and education simultaneously).

Outputs:
  data/processed/dengue_vn_sdi_residuals.csv
        year, ln_cases, sdi, ln_cases_pred_linear, ln_cases_pred_quad,
        residual_linear, residual_quad
  tables/sdi_strip_comparison.csv
        For each climate predictor in raw_data/climate/climate_indices.csv
        (if present), Spearman rho with cases under 4 transformations:
        raw, linear-detrended, first-differenced, SDI-residualised.
        BH-FDR control applied within each strip.

Usage:
  - Always produces residual series.
  - To produce the strip-comparison table, drop a CSV with columns
    [year, index_name, season, value]
    at raw_data/climate/climate_indices.csv. season values: 'annual' or 'MAM'.
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

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "processed"
TABLES = PROJ / "tables"
SDI_FILE = _REPO_ROOT / "raw_data" / "covariates" / "SDI.csv"
CLIMATE_FILE = _REPO_ROOT / "raw_data" / "climate" / "climate_indices.csv"


def spearman_rho(x, y):
    """Spearman rho via rank-Pearson; pure-pandas, no scipy required."""
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values. Returns array same shape."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotone (cumulative min from the right)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def spearman_p_two_sided(rho, n):
    """Two-sided p-value for Spearman rho via t-approximation.
    Valid for n >= 10. Avoids scipy."""
    if not np.isfinite(rho) or n < 4 or abs(rho) >= 1.0:
        return float("nan")
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho * rho))
    # Two-sided Student-t approximation via Cornish-Fisher / standard
    # normal limit (n large). For n>=20 this is within 5% of the exact
    # value, sufficient for FDR ranking.
    z = abs(t) * (1 - 1 / (4 * (n - 2)))
    # Standard normal sf via erfc
    from math import erfc, sqrt
    return float(erfc(z / sqrt(2)))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # ── Load cases + SDI ──────────────────────────────────────────────
    od = pd.read_csv(DATA / "opendengue_vn_annual.csv")[["year", "cases"]]
    sdi = pd.read_csv(SDI_FILE)
    sdi_vn = (sdi[sdi["location_name"] == "Viet Nam"]
              [["year_id", "mean_value"]]
              .rename(columns={"year_id": "year", "mean_value": "sdi"}))

    df = od.merge(sdi_vn, on="year", how="inner").sort_values("year").reset_index(drop=True)
    df["ln_cases"] = np.log(df["cases"])

    # SDI is essentially monotonic over 1990-2023; use both linear and
    # quadratic forms so the user can pick whichever has higher R^2.
    yrs = df["year"].to_numpy()
    sdi_arr = df["sdi"].to_numpy()
    ln = df["ln_cases"].to_numpy()

    coef_lin = np.polyfit(sdi_arr, ln, 1)
    coef_quad = np.polyfit(sdi_arr, ln, 2)
    df["ln_cases_pred_linear"] = np.polyval(coef_lin, sdi_arr)
    df["ln_cases_pred_quad"] = np.polyval(coef_quad, sdi_arr)
    df["residual_linear"] = df["ln_cases"] - df["ln_cases_pred_linear"]
    df["residual_quad"] = df["ln_cases"] - df["ln_cases_pred_quad"]

    # R^2 for the SDI fits (so user can compare to the year-fit baseline)
    ss_tot = float(np.sum((ln - ln.mean()) ** 2))
    r2_lin = 1.0 - float(np.sum(df["residual_linear"] ** 2)) / ss_tot
    r2_quad = 1.0 - float(np.sum(df["residual_quad"] ** 2)) / ss_tot

    # Compare to year-only fit (manuscript v3's linear detrending baseline)
    coef_year = np.polyfit(yrs, ln, 1)
    resid_year = ln - np.polyval(coef_year, yrs)
    r2_year = 1.0 - float(np.sum(resid_year ** 2)) / ss_tot
    df["residual_linear_yearfit"] = resid_year

    df.to_csv(DATA / "dengue_vn_sdi_residuals.csv", index=False)

    print(f"  [05] residual series: {len(df)} years "
          f"({df.year.min()}-{df.year.max()})")
    print(f"  [05] R^2 of ln(cases) ~ year:        {r2_year:.3f}   "
          "(manuscript v3 linear-detrend baseline)")
    print(f"  [05] R^2 of ln(cases) ~ SDI (linear): {r2_lin:.3f}")
    print(f"  [05] R^2 of ln(cases) ~ SDI (quad):   {r2_quad:.3f}")
    if r2_lin > r2_year:
        print(f"  [05] -> SDI explains MORE variance than calendar year, "
              "so SDI-residualisation is a stricter trend correction.")
    else:
        print(f"  [05] -> Year explains slightly more, but SDI is the "
              "domain-informed trend; both should be reported.")

    # ── Strip comparison table (only if climate file present) ─────────
    if CLIMATE_FILE.exists():
        clim = pd.read_csv(CLIMATE_FILE)
        _build_strip_table(df, clim)
    else:
        print(f"  [05] (info) No climate file at {CLIMATE_FILE}.")
        print(f"  [05] To enable strip comparison, save a CSV there with")
        print(f"  [05] columns: year, index_name, season, value")
        print(f"  [05] (season is 'annual' or 'MAM').")
        print(f"  [05] Residual series is saved and ready to feed into")
        print(f"  [05] your existing rank-correlation pipeline.")


def _build_strip_table(df_resid, clim):
    """For each (index_name, season) and lag in {0,1}, compute Spearman
    rho between climate predictor and four versions of cases:
    raw, linear-detrended, first-differenced, SDI-residualised."""
    rows = []
    for (idx, season), block in clim.groupby(["index_name", "season"]):
        block = block.sort_values("year")
        for lag in (0, 1):
            shifted = block.copy()
            shifted["year"] = shifted["year"] + lag
            m = df_resid.merge(shifted[["year", "value"]], on="year", how="inner")
            if len(m) < 10:
                continue
            ln_raw = m["ln_cases"].to_numpy()
            ln_lin = m["residual_linear_yearfit"].to_numpy()
            ln_sdi = m["residual_linear"].to_numpy()
            ln_diff = np.diff(ln_raw)
            x_diff = np.diff(m["value"].to_numpy())
            x = m["value"].to_numpy()
            rows.append(dict(
                index_name=idx, season=season, lag=lag, n=len(m),
                rho_raw=spearman_rho(x, ln_raw),
                rho_lineardetrend=spearman_rho(x, ln_lin),
                rho_firstdiff=spearman_rho(x_diff, ln_diff),
                rho_sdi_residual=spearman_rho(x, ln_sdi),
            ))

    if not rows:
        print("  [05] (warn) climate file present but no usable rows.")
        return
    out = pd.DataFrame(rows)
    # Apply BH-FDR within each strip across the (index_name × season × lag) tests
    for col in ("rho_raw", "rho_lineardetrend", "rho_firstdiff", "rho_sdi_residual"):
        ps = [spearman_p_two_sided(r, n) for r, n in zip(out[col], out["n"])]
        out[col + "_p"] = ps
        out[col + "_pBH"] = bh_fdr(ps)

    out = out.sort_values(["index_name", "season", "lag"]).reset_index(drop=True)
    out.to_csv(TABLES / "sdi_strip_comparison.csv", index=False)
    print(f"  [05] strip comparison: tables/sdi_strip_comparison.csv "
          f"({len(out)} predictor x lag rows)")


def verify():
    df = pd.read_csv(DATA / "dengue_vn_sdi_residuals.csv")
    # residuals should sum (approximately) to zero
    s_lin = df["residual_linear"].sum()
    s_quad = df["residual_quad"].sum()
    print(f"  [verify] residual sums: linear={s_lin:+.4f}  quad={s_quad:+.4f}")
    assert abs(s_lin) < 1e-6, "linear residuals should sum to zero"
    assert abs(s_quad) < 1e-6, "quadratic residuals should sum to zero"
    # SDI should be monotone
    assert df["sdi"].is_monotonic_increasing, "VN SDI should be monotone increasing"
    print("  [verify] residual sums ~ 0; SDI monotone.")


if __name__ == "__main__":
    main()
    verify()
