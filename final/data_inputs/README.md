# data_inputs/ — input CSVs every figure script reads

These are the **processed** CSVs that feed every analysis script. Their
producers are the ingest-layer scripts (00–02) in
[../scripts/](../scripts/) — see [../scripts/README.md](../scripts/README.md)
for the upstream raw sources.

| File | Rows × cols | Schema (key columns) | Years | Used by |
|---|---|---|---|---|
| `opendengue_vn_annual.csv` | 36 × 8 | `year, cases, source_label, source_tier, aggregation, source, n, unit` | 1990 – 2025 | Almost every figure: Fig 1, 2, S1*, S3, S4, S5, S6 (`*S1` reads only climate). Identical content to Table S1. |
| `gbd_dengue_vn.csv` | long-form GBD modelled incidence + DALYs (all ages, all sexes, by measure / metric) | `year, measure_name, metric_name, age_name, sex_name, val, lower, upper, …` | 1990 – 2023 | Fig 1 (DALYs panel), Fig validation. |
| `gbd_dengue_vn_byage.csv` | GBD modelled cases by single-year age × year | `year, age_group, val, …` | 1990 – 2023 | Fig 5 (mean age + age-band shares). |
| `dengue_vn_sdi_residuals.csv` | SDI-residualised log cases series | `year, log_cases, sdi, log_cases_residual, …` | 1990 – 2023 | Input to `08_fig4_fivelens.py` via the strip table (Table S3). |
| `climate_indices.csv` | long form, 8 indices × {annual, MAM} | `year, index_name, season, value`; `index_name ∈ {AMO, DMI, IOBW, MEI, ONI, PDO, SOI, TNA}`, `season ∈ {annual, MAM}` | 1854 – 2025 (analysis filters to 1990 – 2025) | Fig 3, 4, S1, S3, S4, S5, S6. Coverage caveat: AMO ends 2022, DMI ends 2024; the rest run through 2025. |

## Year-range / coverage notes

- **Dengue cases**: 1990–2022 from OpenDengue, 2023–2025 from Vietnamese
  Ministry of Health releases. The single labelled source per year is in
  `source_label`. See Supplement §S1.1 for the merge rule.
- **GBD modelled** cases / DALYs / by-age burden run **only through 2023** —
  the GBD 2023 release is the latest available at submission.
- **AMO coverage**: the NOAA PSL AMO series stops at 2022 in the file
  ingested here (with a partial 2023 month that fails the ≥11-months
  filter). Supplement §S2 "Coverage" paragraph states this explicitly;
  Fig S3 also annotates "n=8" on the two AMO Era C bars.
