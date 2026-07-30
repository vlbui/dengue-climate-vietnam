"""15 - Supplementary Figure S5: descriptive age distribution of dengue burden.

DESCRIPTIVE CONTEXT ONLY (not part of the primary climate analysis). Shows the
share of GBD 2023 modelled dengue incidence by broad age band, 1990-2023,
illustrating that burden is concentrated in children / young adults with a
gradual shift towards older groups over the study period.

Reads : final/data_inputs/gbd_dengue_vn_byage.csv
Writes: final/figures/FigS5_age_distribution.{png,svg}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINAL = Path(__file__).resolve().parents[1]
SRC = FINAL / "data_inputs" / "gbd_dengue_vn_byage.csv"
OUT = FINAL / "figures"

# Lancet-style sequential palette (young -> old)
COLORS = ["#00468B", "#0099B4", "#42B540", "#ED0000"]
BANDS = [("<15 years", 0, 14), ("15-29 years", 15, 29),
         ("30-49 years", 30, 49), ("50+ years", 50, 200)]


def band_of(age_name: str) -> str:
    nums = [int(x) for x in re.findall(r"\d+", age_name)]
    lo = nums[0] if nums else (0 if "<" in age_name else 999)
    for label, blo, bhi in BANDS:
        if blo <= lo <= bhi:
            return label
    return BANDS[-1][0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    inc = df[(df.measure_name == "Incidence") & (df.metric_name == "Number")].copy()
    inc = inc[(inc.year >= 1990) & (inc.year <= 2023)]
    inc["band"] = inc.age_name.map(band_of)
    piv = inc.pivot_table(index="year", columns="band", values="val",
                          aggfunc="sum").fillna(0.0)
    piv = piv[[b for b, _, _ in BANDS]]
    share = piv.div(piv.sum(axis=1), axis=0) * 100

    yrs = share.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.stackplot(yrs, [share[b].to_numpy() for b, _, _ in BANDS],
                 labels=[b for b, _, _ in BANDS], colors=COLORS, alpha=0.92)
    ax.set_xlim(yrs.min(), yrs.max())
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of modelled dengue incidence (%)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "FigS5_age_distribution.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "FigS5_age_distribution.svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # quick log of the shift
    for b, _, _ in BANDS:
        print(f"  [S5] {b:12} 1990={share.loc[1990,b]:5.1f}%  2023={share.loc[2023,b]:5.1f}%")
    print(f"  [S5] figure: {OUT / 'FigS5_age_distribution.png'} + .svg")


if __name__ == "__main__":
    main()
