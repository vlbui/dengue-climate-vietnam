"""_climate_to_csv - build raw_data/climate/climate_indices.csv from per-index files.

Source: raw_data/climate_data/{iobw,mei,dmi,soi,pdo,oni,amo,tna}.csv
  Most: columns [date, year, month, value]
  ONI:  columns [Date, "  ONI from CPC ..."], with missing sentinels {-99.9, -9999.0}

Output: raw_data/climate/climate_indices.csv  (long form)
  columns: [year, index_name, season, value]
  season ∈ {"annual", "MAM"}
    annual = mean of months 1-12 within the year
    MAM    = mean of months 3-5

Quality filter:
  - Drop years where the source has fewer than 11 monthly observations
    (year-level filter; affects both annual and MAM rows for that year).
  - Both means are computed only over non-null months.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2].parent  # d:/gbd
SRC = ROOT / "raw_data" / "climate_data"
DST = ROOT / "raw_data" / "climate" / "climate_indices.csv"

INDICES = ["IOBW", "MEI", "DMI", "SOI", "PDO", "ONI", "AMO", "TNA"]
MIN_MONTHS = 11
ONI_MISSING = (-99.9, -9999.0)


def load_index(name: str) -> pd.DataFrame:
    """Return a DataFrame with columns [year, month, value] for the given index."""
    p = SRC / f"{name.lower()}.csv"
    df = pd.read_csv(p)
    if name == "ONI":
        # ONI has 'Date' and a long column name; recover year, month, value.
        date_col = "Date" if "Date" in df.columns else "date"
        val_col = [c for c in df.columns if c != date_col][0]
        d = pd.to_datetime(df[date_col])
        out = pd.DataFrame({
            "year": d.dt.year.astype(int),
            "month": d.dt.month.astype(int),
            "value": df[val_col].astype(float),
        })
        out.loc[out["value"].isin(ONI_MISSING), "value"] = pd.NA
    else:
        out = df[["year", "month", "value"]].copy()
        out["year"] = out["year"].astype(int)
        out["month"] = out["month"].astype(int)
        out["value"] = out["value"].astype(float)
    return out.dropna(subset=["value"])


def aggregate(name: str, monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly index to annual + MAM long form, year-quality filtered."""
    # year-level quality filter
    n_obs = monthly.groupby("year")["value"].size().rename("n")
    keep_years = n_obs[n_obs >= MIN_MONTHS].index
    m = monthly[monthly["year"].isin(keep_years)]

    # annual = mean across months 1-12
    ann = (
        m.groupby("year")["value"].mean().rename("value").reset_index()
        .assign(index_name=name, season="annual")
    )

    # MAM = mean across months 3-5
    mam_src = m[m["month"].isin([3, 4, 5])]
    # require all 3 months to consider MAM valid for the year
    mam_n = mam_src.groupby("year")["value"].size()
    mam_years = mam_n[mam_n == 3].index
    mam_src = mam_src[mam_src["year"].isin(mam_years)]
    mam = (
        mam_src.groupby("year")["value"].mean().rename("value").reset_index()
        .assign(index_name=name, season="MAM")
    )

    return pd.concat([ann, mam], ignore_index=True)[["year", "index_name", "season", "value"]]


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    pieces = []
    print(f"[climate] reading from {SRC}")
    for name in INDICES:
        monthly = load_index(name)
        long = aggregate(name, monthly)
        ann_n = (long["season"] == "annual").sum()
        mam_n = (long["season"] == "MAM").sum()
        y0, y1 = monthly["year"].min(), monthly["year"].max()
        print(f"  {name:5s}: monthly {len(monthly):>5d} ({y0}-{y1})  "
              f"-> annual={ann_n}  MAM={mam_n}")
        pieces.append(long)

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["index_name", "season", "year"]).reset_index(drop=True)
    out.to_csv(DST, index=False)
    print(f"[climate] wrote {DST}  ({len(out)} rows)")


def verify():
    df = pd.read_csv(DST)
    assert list(df.columns) == ["year", "index_name", "season", "value"], df.columns
    assert set(df["index_name"].unique()) == set(INDICES), df["index_name"].unique()
    assert set(df["season"].unique()) == {"annual", "MAM"}, df["season"].unique()
    assert df["value"].notna().all(), "found NaNs in value"
    # report 1990-onwards coverage per index
    sub = df[df["year"] >= 1990]
    print("[verify] 1990-onwards coverage by (index, season):")
    grid = (
        sub.groupby(["index_name", "season"])["year"]
        .agg(["min", "max", "count"]).reset_index()
        .sort_values(["index_name", "season"])
    )
    for _, r in grid.iterrows():
        print(f"  {r['index_name']:5s} {r['season']:6s}: {int(r['min'])}-{int(r['max'])}  n={int(r['count'])}")


if __name__ == "__main__":
    main()
    verify()
