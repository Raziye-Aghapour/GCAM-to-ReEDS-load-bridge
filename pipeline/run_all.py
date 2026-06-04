"""
run_all.py — End-to-end pipeline driver  (v2 — year-specific shapes)
=====================================================================

Runs the full GCAM → ReEDS load shape pipeline:

  step 1: example_gcam_query.xlsx  → scaling_inputs_MWh.csv          (once)
  step 2: EFS + ResStock + ComStock + EVI-Pro → combined_unscaled_{year}.csv.gz
                                                                  (once per year)
  step 3: combined_{year} + scaling_inputs → scaled_shapes/<scenario>/{year}.csv.gz
                                                                  (once per year)
  step 4: scaled_shapes → <scenario>_load_hourly.h5                   (once)

v2 CHANGE vs v1
---------------
v1 called step 2 once with a single ResStock/ComStock/EVI-Pro file.
v2 calls step 2 in a loop over GCAM years. For each year it looks for:

    {resstock_dir}/resstock_unscaled_texas_{year}.csv.gz
    {comstock_dir}/comstock_unscaled_texas_{year}.csv.gz
    {evipro_dir}/evipro_ldv_unscaled_texas_{year}.csv.gz

These year-specific files come from:
  - dsgrid building shapes notebook v6 → resstock_unscaled_texas_*.csv.gz
                                       → comstock_unscaled_texas_*.csv.gz
  - EVI-Pro notebook v7                → evipro_ldv_unscaled_texas_*.csv.gz

Any or all of --resstock-dir / --comstock-dir / --evipro-dir can be omitted;
the EFS base shape is used unchanged for those subsectors.

ONE-SHOT USAGE:
  python run_all.py \\
      --gcam-xlsx     inputs/example_gcam_query.xlsx \\
      --efs-base      inputs/efs_base_unscaled.csv.gz \\
      --resstock-dir  inputs/ \\
      --comstock-dir  inputs/ \\
      --evipro-dir    inputs/ \\
      --load-factors  inputs/load_factors.csv \\
      --scenario      gcam_default \\
      --years         2025,2030,2035,2040,2045,2050 \\
      --output-h5     outputs/gcam_default_load_hourly.h5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
STEP1 = HERE / "step1_gcam_to_scaling_inputs.py"
STEP2 = HERE / "step2_combine_unscaled_shapes.py"
STEP3 = HERE / "step3_scale_shapes.py"
STEP4 = HERE / "step4_generate_h5.py"

import subprocess


def run(cmd: list, label: str) -> None:
    print(f"\n{'#'*70}\n# {label}\n# {' '.join(str(c) for c in cmd)}\n{'#'*70}")
    r = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if r.returncode != 0:
        print(f"\n!! Step failed: {label}", file=sys.stderr)
        sys.exit(r.returncode)


def infer_year_file(
    directory: Path,
    prefix: str,
    state: str,
    year: int,
) -> Path:
    """Return the expected path for a year-specific shape file."""
    for ext in (".csv.gz", ".csv"):
        p = directory / f"{prefix}_unscaled_{state}_{int(year)}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Year-specific shape file not found for {prefix}/{year} in {directory}.\n"
        f"Expected: {directory / f'{prefix}_unscaled_{state}_{year}.csv.gz'}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gcam-xlsx", required=True,
                   help="Path to example_gcam_query.xlsx")
    p.add_argument("--efs-base", required=True,
                   help="EFS base unscaled CSV (single file, year-agnostic)")
    p.add_argument("--resstock-dir",
                   help="Directory containing resstock_unscaled_texas_{year}.csv.gz files")
    p.add_argument("--comstock-dir",
                   help="Directory containing comstock_unscaled_texas_{year}.csv.gz files")
    p.add_argument("--evipro-dir",
                   help="Directory containing evipro_ldv_unscaled_texas_{year}.csv.gz files")
    p.add_argument("--load-factors", required=True,
                   help="BA share CSV for state→BA disaggregation")
    p.add_argument("--scenario", default="gcam_default",
                   help="Scenario name (default: gcam_default)")
    p.add_argument("--years", default="2025,2030,2035,2040,2045,2050",
                   help="Comma-separated GCAM years")
    p.add_argument("--state", default="texas",
                   help="State name embedded in shape filenames (default: texas)")
    p.add_argument("--output-h5", required=True,
                   help="Path for the final ReEDS H5 output")
    p.add_argument("--workdir", default="pipeline_work",
                   help="Directory for intermediate files (default: pipeline_work/)")
    p.add_argument("--historical-years",
                   help="Comma-separated historical years to zero-pad in the H5")
    p.add_argument("--template-h5",
                   help="Existing template H5 to merge into (merge mode)")
    p.add_argument("--merge-operation", choices=["add", "replace"], default="replace",
                   help="Merge operation when --template-h5 is given")
    p.add_argument("--gcam-scenario", default=None,
                   help=(
                       "If the GCAM xlsx contains multiple scenario rows, "
                       "pass the scenario name to select. Forwarded to step 1 "
                       "as --gcam-scenario. If omitted and multiple GCAM "
                       "scenarios exist, step 1 will raise ValueError."
                   ))
    args = p.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    scaling_inputs = workdir / "scaling_inputs_MWh.csv"
    scaled_dir = workdir / "scaled_shapes"

    # ── Step 1 — once ──────────────────────────────────────────────────────────
    cmd1 = [
        STEP1,
        "--gcam-xlsx", args.gcam_xlsx,
        "--scenario",  args.scenario,
        "--output",    scaling_inputs,
        "--years",     args.years,
    ]
    if args.gcam_scenario:
        cmd1 += ["--gcam-scenario", args.gcam_scenario]
    run(cmd1, "STEP 1: GCAM xlsx → scaling_inputs_MWh.csv")

    # ── Step 2 — once per year ──────────────────────────────────────────────────
    for year in years:
        combined_out = workdir / f"combined_unscaled_{year}.csv.gz"
        cmd2 = [
            STEP2,
            "--efs-base", args.efs_base,
            "--output",   combined_out,
        ]

        if args.resstock_dir:
            try:
                resstock_path = infer_year_file(
                    Path(args.resstock_dir), "resstock", args.state, year)
                cmd2 += ["--resstock", resstock_path]
                print(f"  ResStock {year}: {resstock_path.name}")
            except FileNotFoundError as e:
                print(f"  WARNING: {e}\n  Using EFS base shape for residential {year}.")

        if args.comstock_dir:
            try:
                comstock_path = infer_year_file(
                    Path(args.comstock_dir), "comstock", args.state, year)
                cmd2 += ["--comstock", comstock_path]
                print(f"  ComStock {year}: {comstock_path.name}")
            except FileNotFoundError as e:
                print(f"  WARNING: {e}\n  Using EFS base shape for commercial {year}.")

        if args.evipro_dir:
            try:
                evipro_path = infer_year_file(
                    Path(args.evipro_dir), "evipro_ldv", args.state, year)
                cmd2 += ["--evipro", evipro_path]
                print(f"  EVI-Pro  {year}: {evipro_path.name}")
            except FileNotFoundError as e:
                print(f"  WARNING: {e}\n  Using EFS base shape for LDV {year}.")

        run(cmd2, f"STEP 2 ({year}): Combine unscaled shapes → combined_unscaled_{year}.csv.gz")

    # ── Step 3 — reads per-year combined files ─────────────────────────────────
    run(
        [STEP3,
         "--combined-unscaled-dir", workdir,
         "--scaling-inputs",        scaling_inputs,
         "--output-dir",            scaled_dir,
         "--scenario",              args.scenario,
         "--years",                 args.years],
        "STEP 3: Scale shapes by GCAM annual MWh",
    )

    # ── Step 4 — once, all years ────────────────────────────────────────────────
    cmd4 = [
        STEP4,
        "--scaled-dir",   scaled_dir,
        "--scenario",     args.scenario,
        "--load-factors", args.load_factors,
        "--output",       args.output_h5,
        "--years",        args.years,
    ]
    if args.historical_years:
        cmd4 += ["--historical-years", args.historical_years]
    if args.template_h5:
        cmd4 += ["--template-h5",  args.template_h5,
                 "--operation",    args.merge_operation]
    run(cmd4, "STEP 4: Generate ReEDS H5")

    print(f"\n{'='*70}\nDONE.\n")
    print(f"Intermediate files : {workdir}/")
    print(f"Final H5           : {args.output_h5}")
    out_stem = Path(args.output_h5).stem
    switch_val = out_stem[:-len("_load_hourly")] if out_stem.endswith("_load_hourly") else out_stem
    print(f"\nReEDS cases.csv entry:")
    print(f"   GSw_EFS1_AllYearLoad,{switch_val}")
    print('='*70)


if __name__ == "__main__":
    main()
