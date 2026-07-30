"""01 — Load and tidy the three OpenDengue layers for Vietnam.

Reads the user-prepared OpenDengue extracts under raw_data/opendengue/ and
writes three CSV files with consistent schemas under data/processed/.

Layers:
  * Annual national (1990–2025)  — from annual_dengue_vn_WHOpriority.csv,
    augmented with MoH 2023–2025 figures from manuscript v3.
  * Monthly national (1994–2019) — from monthly_dengue_national_merged.csv.
  * Provincial monthly (1994–2010, 64 provinces) — from spatio_temporal.csv,
    Admin1×Month rows only, TYCHO UUID.

Outputs:
  data/processed/opendengue_vn_annual.csv
  data/processed/opendengue_vn_monthly.csv
  data/processed/opendengue_vn_provincial.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running this script directly from anywhere — add repo root to path
# so `from shared import ...` works without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data" / "processed"
OD = _REPO_ROOT / "raw_data" / "opendengue"

# Manuscript v3 reports MoH-sourced figures for 2023–2025; the
# annual_dengue_vn_WHOpriority.csv file currently stops at 2022.
# These three rows are added with explicit source provenance.
MOH_2023_2025 = pd.DataFrame([
    {"year": 2023, "cases": 172000, "source_label": "MOH_VNM_pubcomm",
     "source_tier": 6, "aggregation": "MoH public communication",
     "source": "MOH-VNM-2023", "n": 1, "unit": "year"},
    {"year": 2024, "cases": 148237, "source_label": "MOH_VNM_pubcomm",
     "source_tier": 6, "aggregation": "MoH public communication (derived 190040/1.282)",
     "source": "MOH-VNM-2024", "n": 1, "unit": "year"},
    {"year": 2025, "cases": 190040, "source_label": "MOH_VNM_pubcomm",
     "source_tier": 6, "aggregation": "MoH public communication",
     "source": "MOH-VNM-2025", "n": 1, "unit": "year"},
])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Layer 1: annual national ──────────────────────────────────────
    ann = pd.read_csv(OD / "annual_dengue_vn_WHOpriority.csv")
    if ann["year"].max() < 2025:
        ann = pd.concat([ann, MOH_2023_2025], ignore_index=True)
    ann = ann.sort_values("year").reset_index(drop=True)
    ann.to_csv(OUT / "opendengue_vn_annual.csv", index=False)
    print(f"  [01] annual:      {len(ann):>5} rows  {ann.year.min()}–{ann.year.max()}")

    # ── Layer 2: monthly national ─────────────────────────────────────
    mon = pd.read_csv(OD / "monthly_dengue_national_merged.csv")
    mon["ym"] = pd.to_datetime(mon["ym"])
    mon = mon.sort_values("ym").reset_index(drop=True)
    mon.to_csv(OUT / "opendengue_vn_monthly.csv", index=False)
    print(f"  [01] monthly:     {len(mon):>5} rows  {mon.year.min()}–{mon.year.max()}")

    # ── Layer 3: provincial monthly ───────────────────────────────────
    st = pd.read_csv(OD / "spatio_temporal.csv")
    prov = st[(st["S_res"] == "Admin1") & (st["T_res"] == "Month")].copy()
    prov["calendar_start_date"] = pd.to_datetime(prov["calendar_start_date"])
    prov["calendar_end_date"] = pd.to_datetime(prov["calendar_end_date"])
    prov = prov.rename(columns={
        "adm_1_name": "province",
        "Year": "year",
        "dengue_total": "cases",
    })
    prov["month"] = prov["calendar_start_date"].dt.month
    prov = prov[["province", "year", "month", "cases",
                 "calendar_start_date", "calendar_end_date",
                 "case_definition_standardised", "UUID"]]
    prov = prov.sort_values(["province", "year", "month"]).reset_index(drop=True)
    prov.to_csv(OUT / "opendengue_vn_provincial.csv", index=False)
    n_prov = prov["province"].nunique()
    print(f"  [01] provincial:  {len(prov):>5} rows  "
          f"{prov.year.min()}–{prov.year.max()}  ({n_prov} provinces)")


def verify():
    """Spot-check key totals against manuscript v3 reported numbers."""
    ann = pd.read_csv(OUT / "opendengue_vn_annual.csv")
    a90 = ann.loc[ann.year == 1990, "cases"].iloc[0]
    a98 = ann.loc[ann.year == 1998, "cases"].iloc[0]
    a22 = ann.loc[ann.year == 2022, "cases"].iloc[0]
    a25 = ann.loc[ann.year == 2025, "cases"].iloc[0]
    print(f"  [verify] 1990={a90:,.0f} (mss: 37,569)  "
          f"1998={a98:,.0f} (mss: 234,866)  "
          f"2022={a22:,.0f} (mss: 367,729)  "
          f"2025={a25:,.0f} (mss: 190,040)")
    assert int(a90) == 37569
    assert int(a98) == 234866
    assert int(a22) == 367729
    assert int(a25) == 190040
    print("  [verify] all annual sentinel values match manuscript v3")


if __name__ == "__main__":
    main()
    verify()
