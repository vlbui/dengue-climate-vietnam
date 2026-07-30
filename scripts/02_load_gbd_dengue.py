"""02 — Load and slice GBD 2023 dengue burden for Vietnam.

Reads raw_data/burden/dengue_vn.csv (IHME GBD 2023 query, 1980–2023, all
measures × all metrics × 17 age groups + All ages + Age-standardized,
Both/Male/Female) and writes two clean CSVs:

  * data/processed/gbd_dengue_vn.csv
        long form, Both sex, All ages + Age-standardized only
  * data/processed/gbd_dengue_vn_byage.csv
        long form, Both sex, 5-year age groups (no aggregates)

Manuscript v3 covers 1990 onward; we keep 1980–1989 as a pre-period for
reference but downstream analysis filters to 1990+.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data" / "processed"
RAW = _REPO_ROOT / "raw_data" / "burden" / "dengue_vn.csv"

KEEP_COLS = ["year", "sex_name", "age_name", "measure_name",
             "metric_name", "val", "lower", "upper"]

# Aggregate age categories from IHME — keep separate from per-age slice.
AGGREGATE_AGES = {"All ages", "Age-standardized"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW)

    # Both-sex only for our analyses; sex stratification is a future
    # follow-up (manuscript v3 does not stratify by sex).
    both = df[df["sex_name"] == "Both"].copy()

    # Slice 1: All ages + Age-standardized aggregates (used for trends).
    agg = both[both["age_name"].isin(AGGREGATE_AGES)].copy()
    agg = agg[KEEP_COLS].sort_values(
        ["measure_name", "metric_name", "age_name", "year"]
    ).reset_index(drop=True)
    agg.to_csv(OUT / "gbd_dengue_vn.csv", index=False)
    print(f"  [02] aggregate ages: {len(agg):>5} rows  "
          f"({agg.year.min()}–{agg.year.max()})")
    _summarise(agg, "aggregate")

    # Slice 2: 5-year age groups only (used for age-shift sub-analysis).
    by_age = both[~both["age_name"].isin(AGGREGATE_AGES)].copy()
    by_age = by_age[KEEP_COLS].sort_values(
        ["measure_name", "metric_name", "age_name", "year"]
    ).reset_index(drop=True)
    by_age.to_csv(OUT / "gbd_dengue_vn_byage.csv", index=False)
    n_age = by_age["age_name"].nunique()
    print(f"  [02] by 5-yr age:    {len(by_age):>5} rows  ({n_age} age groups)")
    _summarise(by_age, "by-age")


def _summarise(df, label):
    n_meas = df["measure_name"].nunique()
    n_metric = df["metric_name"].nunique()
    print(f"           [{label}] {n_meas} measure(s), {n_metric} metric(s):  "
          f"{', '.join(sorted(df['measure_name'].unique()))}")


def verify():
    """Sanity checks against manuscript v3 sentinel observations."""
    agg = pd.read_csv(OUT / "gbd_dengue_vn.csv")

    # Pull GBD all-ages incidence numbers for 1990, 1998, 2022 to sanity
    # check magnitude vs OpenDengue (they shouldn't match exactly, but
    # should be in the same order of magnitude).
    inc = agg[(agg.measure_name == "Incidence")
              & (agg.metric_name == "Number")
              & (agg.age_name == "All ages")].set_index("year")
    print(f"  [verify] GBD all-ages dengue incidence (Number):")
    for y in (1990, 1998, 2010, 2022):
        if y in inc.index:
            v = inc.loc[y, "val"]
            lo = inc.loc[y, "lower"]
            hi = inc.loc[y, "upper"]
            print(f"           {y}: {v:>13,.0f}  [{lo:>13,.0f}–{hi:>13,.0f}]")

    asr = agg[(agg.measure_name == "DALYs (Disability-Adjusted Life Years)")
              & (agg.metric_name == "Rate")
              & (agg.age_name == "Age-standardized")].set_index("year")
    print(f"  [verify] GBD age-std DALY rate per 100k:")
    for y in (1990, 2000, 2010, 2022):
        if y in asr.index:
            print(f"           {y}: {asr.loc[y, 'val']:>7.2f}")


if __name__ == "__main__":
    main()
    verify()
