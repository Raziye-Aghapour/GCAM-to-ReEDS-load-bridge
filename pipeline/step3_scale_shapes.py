"""
step3_scale_shapes.py
=====================

Apply per-state, per-year, per-subsector_group scaling to combined unscaled
shape CSVs using scaling_inputs_MWh.csv from step 1.

v2 CHANGE vs v1
---------------
v1 accepted a single --combined-unscaled file (one combined CSV for all years).
v2 accepts --combined-unscaled-dir pointing to a directory that contains one
combined CSV per GCAM year, named:

    combined_unscaled_{year}.csv.gz

This matches the output layout of running step 2 once per year (which is
required because ResStock, ComStock, and EVI-Pro shapes now vary by year due
to GCAM technology-weighted blending in the dsgrid and EVI-Pro builder
notebooks).

The --combined-unscaled flag from v1 is still supported for backwards
compatibility: if a single file is passed, that same file is used for
every year (v1 behaviour).

Scaling logic (unchanged from v1):
  For each (scenario, year, subsector_group):
    1. Filter combined CSV to rows whose subsector is in the comma-split list.
    2. Sum across subsectors per state to get per-state unscaled annual MWh.
    3. Compute ratio = target / unscaled_sum; scale hourly values.
    4. Zero-to-positive: if unscaled_sum=0 but target>0, spread uniformly.

Output schema:
  scaled_shapes/<scenario>/<year>.csv.gz
  Columns: sector, subsector, weather_datetime, alabama, ..., wyoming

Usage (v2 — year-specific combined files):
  python step3_scale_shapes.py \\
      --combined-unscaled-dir  pipeline_work/ \\
      --scaling-inputs         pipeline_work/scaling_inputs_MWh.csv \\
      --output-dir             pipeline_work/scaled_shapes \\
      --scenario               gcam_default

Usage (v1 backwards-compatible — single combined file):
  python step3_scale_shapes.py \\
      --combined-unscaled  pipeline_work/combined_unscaled.csv.gz \\
      --scaling-inputs     pipeline_work/scaling_inputs_MWh.csv \\
      --output-dir         pipeline_work/scaled_shapes \\
      --scenario           gcam_default
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

UCS_STATE_COLUMNS = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "district of columbia", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
]

REQUIRED_UNSCALED_COLS = ["sector", "subsector", "weather_datetime"] + UCS_STATE_COLUMNS


def load_unscaled(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    df = pd.read_csv(path, compression=compression)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_UNSCALED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Unscaled CSV {path.name} missing columns: {missing[:5]}")
    df["weather_datetime"] = pd.to_datetime(df["weather_datetime"])
    return df[REQUIRED_UNSCALED_COLS].copy()


def resolve_unscaled_path(
    combined_unscaled_dir: Optional[Path],
    combined_unscaled_file: Optional[Path],
    year: int,
) -> Path:
    """Return the path to the combined unscaled CSV for a given year.

    v2: looks for combined_unscaled_{year}.csv.gz in combined_unscaled_dir.
    v1 fallback: returns combined_unscaled_file directly (same file for all years).
    """
    if combined_unscaled_dir is not None:
        candidate = combined_unscaled_dir / f"combined_unscaled_{int(year)}.csv.gz"
        if candidate.exists():
            return candidate
        candidate_csv = combined_unscaled_dir / f"combined_unscaled_{int(year)}.csv"
        if candidate_csv.exists():
            return candidate_csv
        raise FileNotFoundError(
            f"No combined unscaled file for year {year} in {combined_unscaled_dir}.\n"
            f"Expected: {candidate}\n"
            f"Run step 2 with --output pipeline_work/combined_unscaled_{year}.csv.gz"
        )
    if combined_unscaled_file is not None:
        if not combined_unscaled_file.exists():
            raise FileNotFoundError(
                f"Combined unscaled file not found: {combined_unscaled_file}"
            )
        return combined_unscaled_file
    raise ValueError("Must provide either --combined-unscaled-dir or --combined-unscaled")


def report_subsector_contract(
    unscaled_df: pd.DataFrame,
    scaling_inputs: pd.DataFrame,
    label: str = "",
) -> None:
    """Verify every subsector string in scaling_inputs.subsector_group is present
    in unscaled_df.subsector. Fails fast with a clear message."""
    unscaled_subs = set(unscaled_df["subsector"].unique())
    referenced = set()
    for sg in scaling_inputs["subsector_group"].unique():
        for piece in str(sg).split(","):
            piece = piece.strip()
            if piece:
                referenced.add(piece)

    missing = referenced - unscaled_subs
    if missing:
        raise ValueError(
            f"scaling_inputs references {len(missing)} subsector(s) "
            f"not in {label or 'unscaled CSV'}:\n  "
            + "\n  ".join(sorted(missing))
            + "\n\nFix: check that step 2 produced a combined unscaled CSV "
            "that includes these subsector strings."
        )

    dangling = unscaled_subs - referenced
    if dangling:
        print(f"  NOTE: {len(dangling)} unscaled subsector(s) not referenced by "
              "scaling_inputs will be DROPPED from scaled output:")
        for s in sorted(dangling):
            print(f"    - {s}")
    print(f"  Contract OK: all {len(referenced)} required subsectors present{' in '+label if label else ''}")


def scale_for_year(
    unscaled_df: pd.DataFrame,
    scaling_inputs: pd.DataFrame,
    scenario: str,
    year: int,
) -> pd.DataFrame:
    """Return the scaled shape DataFrame for (scenario, year)."""
    sel = scaling_inputs[
        (scaling_inputs["scenario"] == scenario)
        & (scaling_inputs["year"] == int(year))
    ].copy()
    if sel.empty:
        raise ValueError(
            f"No scaling_inputs rows for scenario={scenario!r}, year={year}.\n"
            f"Available scenarios: {sorted(scaling_inputs['scenario'].unique())}\n"
            f"Available years: {sorted(scaling_inputs['year'].unique())}"
        )

    unscaled_subs = set(unscaled_df["subsector"].unique())
    state_cols = [c for c in UCS_STATE_COLUMNS if c in unscaled_df.columns]
    scaled_pieces: List[pd.DataFrame] = []

    for _, sg_row in sel.iterrows():
        group_str = str(sg_row["subsector_group"]).strip()
        members = [s.strip() for s in group_str.split(",") if s.strip()]
        members_present = [s for s in members if s in unscaled_subs]
        if not members_present:
            print(f"  SKIP group {group_str!r}: no member subsectors in unscaled CSV")
            continue

        group_df = unscaled_df[unscaled_df["subsector"].isin(members_present)].copy()
        group_sum = group_df[state_cols].sum()

        for st in state_cols:
            target = float(sg_row[st]) if st in sg_row.index else 0.0
            unscaled_sum = float(group_sum[st])
            if unscaled_sum == 0 and target > 0:
                per_hour = target / len(group_df)
                group_df[st] = float(per_hour)
            elif unscaled_sum > 0:
                group_df[st] = group_df[st] * (target / unscaled_sum)
            else:
                group_df[st] = 0.0

        scaled_pieces.append(group_df)

    if not scaled_pieces:
        raise RuntimeError(f"No scaled output produced for {scenario}/{year}")
    scaled = pd.concat(scaled_pieces, ignore_index=True)
    scaled = scaled.sort_values(["sector", "subsector", "weather_datetime"]).reset_index(drop=True)
    return scaled


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--combined-unscaled-dir",
        help=(
            "[v2] Directory containing combined_unscaled_{year}.csv.gz files "
            "(one per GCAM year, produced by running step 2 once per year)."
        ),
    )
    p.add_argument(
        "--combined-unscaled",
        help=(
            "[v1 compat] Path to a single combined unscaled CSV used for all years. "
            "Use --combined-unscaled-dir instead when shapes vary by year."
        ),
    )
    p.add_argument("--scaling-inputs", required=True,
                   help="Path to scaling_inputs_MWh.csv from step 1")
    p.add_argument("--output-dir", default="scaled_shapes",
                   help="Directory for <scenario>/<year>.csv.gz outputs")
    p.add_argument("--scenario",
                   help="Scenario to process (default: all in scaling_inputs)")
    p.add_argument("--years",
                   help="Comma-separated years (default: all in scaling_inputs)")
    args = p.parse_args()

    if args.combined_unscaled_dir is None and args.combined_unscaled is None:
        p.error("Provide --combined-unscaled-dir (v2) or --combined-unscaled (v1 compat)")

    combined_dir = Path(args.combined_unscaled_dir) if args.combined_unscaled_dir else None
    combined_file = Path(args.combined_unscaled) if args.combined_unscaled else None

    print(f"Loading scaling_inputs from {args.scaling_inputs}")
    scaling = pd.read_csv(args.scaling_inputs)
    for col in ("scenario", "subsector_group", "year"):
        if col not in scaling.columns:
            raise ValueError(f"scaling_inputs must have column '{col}'. Found: {list(scaling.columns)}")
    print(f"  Shape: {scaling.shape}")
    print(f"  Scenarios: {sorted(scaling['scenario'].unique())}")
    print(f"  Years: {sorted(scaling['year'].unique())}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [args.scenario] if args.scenario else sorted(scaling["scenario"].unique())
    years = (
        [int(y.strip()) for y in args.years.split(",") if y.strip()]
        if args.years
        else sorted(scaling["year"].unique())
    )

    for sc in scenarios:
        sc_dir = out_dir / sc
        sc_dir.mkdir(parents=True, exist_ok=True)

        for yr in years:
            unscaled_path = resolve_unscaled_path(combined_dir, combined_file, yr)
            print(f"\nYear {yr}: loading unscaled from {unscaled_path.name}")
            unscaled = load_unscaled(unscaled_path)
            print(f"  Shape: {unscaled.shape}, {unscaled['subsector'].nunique()} subsectors")

            print(f"  Validating subsector contract for year {yr}...")
            report_subsector_contract(
                unscaled,
                scaling[scaling["scenario"] == sc],
                label=unscaled_path.name,
            )

            print(f"  Scaling scenario={sc!r}, year={yr}...")
            scaled = scale_for_year(unscaled, scaling, sc, int(yr))

            out_path = sc_dir / f"{int(yr)}.csv.gz"
            scaled.to_csv(out_path, index=False, compression="gzip")

            state_cols = [c for c in UCS_STATE_COLUMNS if c in scaled.columns]
            total = scaled[state_cols].sum().sum()
            target_total = scaling[
                (scaling["scenario"] == sc) & (scaling["year"] == int(yr))
            ][state_cols].sum().sum()
            err = abs(total - target_total) / target_total if target_total > 0 else 0.0
            print(f"  Wrote {out_path.name}: "
                  f"total={total:,.1f} MWh, target={target_total:,.1f} MWh, "
                  f"rel error={err:.2e}")

    print(f"\nAll scaled shapes written to {out_dir}/")


if __name__ == "__main__":
    main()
