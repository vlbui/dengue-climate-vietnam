# final/ — review bundle for the Vietnam dengue × climate manuscript

This directory is the review bundle for the manuscript and supplement. In the
**code release** only the scripts and these index READMEs are committed — the
figures, tables, input CSVs and manuscript documents are not (see the root
`README.md` for why, and for how to regenerate them). The tables below still
describe those artefacts in full, so the bundle remains documented end to end;
filenames appear as plain text because the files are not committed here.

## Sub-directories

| Folder | Holds | One-line index |
|---|---|---|
| `scripts/` | The 16 Python scripts that generate every figure and every Supplementary Table. Numbered in pipeline order: 00 = climate ingest, 01–03 = data loaders / validation, 05 / 07 = derivation, 04 / 06 / 08–10 = main-text figures, 11–15 = supplement figures, 16 = robustness. **Committed.** | [scripts/README.md](scripts/README.md) |
| `figures/` | All 10 chart figures (PNG + SVG) embedded in the two documents (5 main + 5 supp). *Not committed.* | [figures/README.md](figures/README.md) |
| `tables/` | The 6 Supplementary Tables as CSVs, plus a `_supporting/` folder with intermediate tables that feed the figures. *Not committed.* | [tables/README.md](tables/README.md) |
| `data_inputs/` | The 5 processed input CSVs the scripts read. *Not committed.* | [data_inputs/README.md](data_inputs/README.md) |
| `manuscript/` | The manuscript and supplement `.docx` documents. *Not committed.* | — |

## How to navigate the bundle

- **"What generated this figure in the paper?"** — go to [figures/README.md](figures/README.md); each row maps a figure file to its script and its input data.
- **"What data is behind this Supplementary Table?"** — see [tables/README.md](tables/README.md). Each table CSV has the same row schema as the Word table.
- **"How is the analysis computed end-to-end?"** — read [scripts/README.md](scripts/README.md) top to bottom. Scripts are numbered in pipeline order; the README spells out inputs / outputs per script.

## Visual style of every figure

All 10 chart figures share one canonical style copied from the main-text
generators (`04_fig1_reported_cases.py` and `09_fig2_descriptive.py`):

- Colour palette (Lancet hex): `LANCET_BLUE="#00468B"`, `LANCET_RED="#ED0000"`,
  `LANCET_GREEN="#42B540"`, `LANCET_INK="#1B1919"`, `LANCET_GREY="#ADB6B6"`,
  `LANCET_TEAL="#0099B4"`.
- No figure-level title. Panel labels small, left-aligned, not bold:
  `ax.set_title("(A) ...", loc="left", fontsize=9.5, color=LANCET_INK)`.
- Grid `ax.grid(True, ls="--", alpha=0.3)`.
- Saved at `dpi=300, bbox_inches="tight", facecolor="white"` as both `.png`
  and `.svg`.

## Outbreak-year definition (single source of truth)

Across the manuscript and supplement an "outbreak year" is a year whose
detrended log-case z-score exceeds +1.5 SD. On the current data this
gives **two outbreak years: 1998 and 2022**. The same definition is used
in main-text Figure 2 (red markers + caption), in Supplement Fig S3 (red
markers + caption), and in the main-text Methods.

(An earlier draft used a top-5 ranking definition for the S5 ROC
analysis, which gave 5 "outbreak" years domin