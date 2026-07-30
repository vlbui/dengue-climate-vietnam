"""run_all - end-to-end pipeline driver for project 13.

Imports the six numbered modules and runs each one's main() + verify().
Resilient: if a step fails, prints the traceback and continues to the next.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THIS = Path(__file__).resolve().parent
STEPS = [
    "01_load_opendengue.py",
    "02_load_gbd_dengue.py",
    "03_validate_opendengue_vs_gbd.py",
    "04_burden_3panel.py",
    "05_sdi_residualise.py",
    "06_age_shift.py",
    "07_ar_prewhiten.py",
]


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("=" * 64)
    print(" project 13 / dengue x climate x GBD - run_all")
    print("=" * 64)
    failures = []
    for step in STEPS:
        path = THIS / step
        print(f"\n>>> {step}")
        try:
            mod = _import(path)
            if hasattr(mod, "main"):
                mod.main()
            if hasattr(mod, "verify"):
                mod.verify()
        except Exception:
            failures.append(step)
            print(f"  [FAIL] {step}")
            traceback.print_exc()
    print("\n" + "=" * 64)
    if failures:
        print(f" DONE with {len(failures)} failure(s): {failures}")
        sys.exit(1)
    print(f" DONE - all {len(STEPS)} steps completed and verified.")


if __name__ == "__main__":
    main()
