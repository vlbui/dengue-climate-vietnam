# Vietnam dengue × climate — analysis code

Analysis code for the manuscript *"Parallel trends inflate climate–dengue
correlations: 36 years of evidence from Vietnam"* (Vietnam dengue × climate ×
GBD, 1990–2025).

## Scope of this repository

**Code only.** Data, figures, tables and manuscript documents are deliberately
not committed:

| Not included here | Where it lives |
|---|---|
| Input and processed data | Public sources — see **Data sources** below; regenerate with scripts `00`–`02` |
| Figures (PNG / SVG) | Reproduced by the figure scripts; final versions appear in the paper |
| Supplementary tables (CSV) | Reproduced by the analysis scripts |
| Manuscript and supplement | Published with the paper, not part of the code release |

The READMEs under `final/` still document those artefacts — schemas, figure
captions, per-script inputs and outputs — so the pipeline is described end to
end even though the artefacts themselves are not in this repository. Filenames
in those tables are written as plain text rather than links, because the files
they name are not committed here.

## Layout

```
.
├── final/
│   ├── scripts/        ← the pipeline as submitted (00–16)
│   ├── data_inputs/    ← README only: schema + provenance of each input CSV
│   ├── figures/        ← README only: caption + generating script per figure
│   └── tables/         ← README only: schema + source script per table
└── scripts/            ← earlier working pipeline, kept for provenance
```

`final/scripts/` is the canonical pipeline: 16 numbered steps that produce every
figure and every Supplementary Table in the submitted manuscript. The top-level
`scripts/` directory is the earlier iteration of the same analysis, retained so
the development history of the results is auditable; it is not the version the
paper reports.

## Data sources

All inputs are public. None are redistributed here.

| Source | Used for |
|---|---|
| [OpenDengue](https://opendengue.org/) | Vietnam annual reported dengue cases, 1990–2022 |
| Vietnamese Ministry of Health releases | Reported cases, 2023–2025 |
| [GBD 2023](https://vizhub.healthdata.org/gbd-results/) (IHME) | Modelled dengue incidence and DALYs, all-ages and by-age, 1990–2023; population denominators |
| [NOAA PSL climate indices](https://psl.noaa.gov/data/climateindices/list/) | AMO, DMI, IOBW, MEI, ONI, PDO, SOI, TNA |
| GBD Socio-demographic Index (SDI) | SDI-residualisation |

GBD data are subject to the IHME licence terms and must be downloaded directly
from IHME.

## Reproducing

```bash
pip install numpy pandas matplotlib scipy statsmodels
```

Download the sources above, then run the pipeline in order:

```bash
# ingest
python final/scripts/00_climate_to_csv.py
python final/scripts/01_load_opendengue.py
python final/scripts/02_load_gbd_dengue.py
python final/scripts/03_validate_opendengue_vs_gbd.py

# derivation
python final/scripts/05_sdi_residualise.py
python final/scripts/07_ar_prewhiten.py

# figures and robustness
python final/scripts/04_fig1_reported_cases.py
python final/scripts/06_fig5_standardized_timeseries.py
python final/scripts/08_fig4_fivelens.py
python final/scripts/09_fig2_descriptive.py
python final/scripts/10_fig3_raw_correlations.py
python final/scripts/11_figS1_climate_panel.py
python final/scripts/12_figS2_era_stratified.py
python final/scripts/13_figS3_iobw_scatter.py
python final/scripts/14_figS4_iobw_slope_specs.py
python final/scripts/15_figS5_age_distribution.py
python final/scripts/16_ar_bootstrap_all_survivors.py
```

Numbered prefixes are pipeline order, not strict execution order — most scripts
read only their immediate inputs and can be run independently once those exist.
See `final/scripts/README.md` for the inputs and outputs of each step.

Scripts resolve their paths relative to the repository layout and expect the
data directories to exist; create them and populate them from the sources above
before running.

## Determinism

All bootstrap procedures use `numpy.random.default_rng(20260524)` with
`n_boot = 5000`, so the reported figures are exactly reproducible.

## Outbreak-year definition

An "outbreak year" is a year whose detrended log-case z-score exceeds +1.5 SD.
On the current data that gives two outbreak years, **1998 and 2022**. The same
definition is used in main-text Figure 2, in Supplement Figure S3, and in the
Methods.
