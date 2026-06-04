"""
step1_gcam_to_scaling_inputs.py
================================

Read example_gcam_query.xlsx and produce scaling_inputs_MWh.csv with the
11-group EFS subsector_group ontology we settled on previously.

GCAM xlsx schema (locked from prior work):
  - Sheets: 'Buildling' (sic — note typo in original Zenodo file),
            'Transportation', 'Industry'
  - Each sheet has columns: region, sector, subsector, input, scenario,
                            then year columns (1990, 1995, ..., 2050)
  - We filter to input == 'electricity' and ignore non-USA regions if
    present (some queries return USA totals; only state-level rows are used).

Output schema (matches main.py contract):
  scenario, subsector_group, year, alabama, alaska, ..., wyoming
  - All 51 states + DC in lowercase full names
  - Year column dtype is int
  - Values are float MWh

Hierarchical-rollup handling for Transportation:
  GCAM nests trn_pass > trn_pass_road > trn_pass_road_LDV > trn_pass_road_LDV_4W.
  Summing all of these double-counts. We sum only the LEAF rows for each
  EFS group, using the explicit GCAM_TO_EFS_GROUP map (which only includes
  leaves for transportation).

Usage:
  python step1_gcam_to_scaling_inputs.py \
      --gcam-xlsx /path/to/example_gcam_query.xlsx \
      --scenario gcam_default \
      --output scaling_inputs_MWh.csv \
      [--years 2025,2030,2035,2040,2045,2050]
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# 51 lowercase full state names — must match main.py / build_reeds_h5.py
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

# US 2-letter to full lowercase
US_ABBR_TO_NAME = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
    "DC": "district of columbia", "FL": "florida", "GA": "georgia", "HI": "hawaii",
    "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina",
    "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
    "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming",
}

# Canonical sheet names. We tolerate the 'Buildling' typo in the original
# Zenodo xlsx and the corrected 'Buildings' if someone has a fixed version.
SHEET_TO_FAMILY = {
    "Buildling": "buildings",
    "Buildings": "buildings",
    "Building":  "buildings",
    "Transportation": "transportation",
    "Industry":       "industrial",
    "Industrial":     "industrial",
}

# GCAM end-use → EFS subsector_group, locked from the previous chat.
# Buildings: 24 GCAM end-uses → 7 EFS groups
# Transportation: maps only LEAVES (not roll-ups like trn_pass / trn_pass_road)
# Industry: GCAM has only 1 row → 1 EFS group
GCAM_TO_EFS_GROUP: Dict[Tuple[str, str], str] = {
    # --- residential ---
    ("buildings", "resid heating"):           "residential space heating and cooling",
    ("buildings", "resid cooling"):           "residential space heating and cooling",
    ("buildings", "resid hot water"):         "residential water heating",
    ("buildings", "resid clothes dryers"):    "residential clothes and dish washing/drying",
    ("buildings", "resid clothes washers"):   "residential clothes and dish washing/drying",
    ("buildings", "resid dishwashers"):       "residential clothes and dish washing/drying",
    ("buildings", "resid lighting"):          "residential other",
    ("buildings", "resid computers"):         "residential other",
    ("buildings", "resid cooking"):           "residential other",
    ("buildings", "resid freezers"):          "residential other",
    ("buildings", "resid furnace fans"):      "residential other",
    ("buildings", "resid refrigerators"):     "residential other",
    ("buildings", "resid televisions"):       "residential other",
    ("buildings", "resid other"):             "residential other",

    # --- commercial ---
    ("buildings", "comm heating"):     "commercial space heating and cooling",
    ("buildings", "comm cooling"):     "commercial space heating and cooling",
    ("buildings", "comm hot water"):   "commercial water heating",
    ("buildings", "comm lighting"):    "commercial other",
    ("buildings", "comm cooking"):     "commercial other",
    ("buildings", "comm refrigeration"): "commercial other",
    ("buildings", "comm ventilation"): "commercial space heating and cooling",  # FIX: dsgrid notebook maps fans/heat_recovery/heat_rejection/pumps → space H&C (line 200,203-206)
    ("buildings", "comm office"):      "commercial other",
    ("buildings", "comm non-building"):"commercial other",
    ("buildings", "comm other"):       "commercial other",

    # --- transportation: only leaves ---
    # The xlsx contains hierarchical rollups (trn_pass, trn_pass_road, etc.) that
    # we deliberately exclude to avoid double-counting.
    ("transportation", "trn_pass_road_LDV_4W"):  "transportation light-duty vehicles",
    ("transportation", "trn_freight_road"):      "transportation medium+heavy-duty trucks",
    ("transportation", "trn_aviation_intl"):     "transportation other",
    ("transportation", "trn_shipping_intl"):     "transportation other",

    # --- industrial ---
    ("industrial", "other industrial energy use"): "industrial (aggregate)",
}

# Subsector_group → list of subsector strings that the unscaled-shape CSV
# will use. main.py splits subsector_group on commas and matches each part
# against the 'subsector' column. So a group with multiple subsectors like
# "industrial machine drives, industrial other, industrial process heat"
# means the unscaled CSV must have ALL three subsector strings present.
EFS_GROUP_TO_SUBSECTORS: Dict[str, List[str]] = {
    "residential space heating and cooling":       ["residential space heating and cooling"],
    "residential water heating":                   ["residential water heating"],
    "residential clothes and dish washing/drying": ["residential clothes and dish washing/drying"],
    "residential other":                           ["residential other"],
    "commercial space heating and cooling":        ["commercial space heating and cooling"],
    "commercial water heating":                    ["commercial water heating"],
    "commercial other":                            ["commercial other"],
    "industrial (aggregate)":                      ["industrial machine drives", "industrial other", "industrial process heat"],
    "transportation light-duty vehicles":          ["transportation light-duty vehicles"],
    "transportation medium+heavy-duty trucks":     ["transportation medium-duty trucks", "transportation heavy-duty trucks"],
    "transportation other":                        ["transportation other"],
}


def normalize_state(region: str) -> Optional[str]:
    """Convert GCAM 'region' value to UCS lowercase full state name.

    GCAM-USA 'region' uses 2-letter codes (TX, CA). Some queries also return
    'USA' as a country aggregate; those rows are dropped.
    """
    if not isinstance(region, str):
        return None
    r = region.strip()
    if r.upper() == "USA":
        return None
    if r.upper() in US_ABBR_TO_NAME:
        return US_ABBR_TO_NAME[r.upper()]
    if r.lower() in UCS_STATE_COLUMNS:
        return r.lower()
    return None


def load_gcam_xlsx(path: Path, unit_to_mwh: float = 2.7777778e8) -> pd.DataFrame:
    """Read all relevant sheets from example_gcam_query.xlsx, return long-format frame.

    Output columns: state (lowercase), sector_family, gcam_subsector, year (int), mwh (float)

    Args:
        path: Path to the GCAM xlsx file.
        unit_to_mwh: Conversion factor from the xlsx value units to MWh.
            Default 2.7777778e8 assumes values are in EJ (1 EJ = 277,777,778 MWh).
            Use 1e6 for TWh, or 1.0 if values are already in MWh.
            Set via --input-unit on the CLI.
    """
    if not path.exists():
        raise FileNotFoundError(f"GCAM xlsx not found: {path}")

    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True)
    sheets_present = list(wb.sheetnames)
    print(f"Sheets in xlsx: {sheets_present}")

    rows = []
    for sheet in sheets_present:
        family = SHEET_TO_FAMILY.get(sheet)
        if family is None:
            print(f"  Skipping unrecognized sheet: {sheet!r}")
            continue

        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]

        # Filter to electricity input
        if "input" in df.columns:
            df = df[df["input"].astype(str).str.lower().str.strip() == "electricity"].copy()
        else:
            print(f"  WARN: sheet {sheet!r} has no 'input' column; using all rows")

        if df.empty:
            print(f"  Sheet {sheet!r}: 0 electricity rows")
            continue

        # Identify the per-end-use label column. GCAM xlsx outputs use
        # 'subsector' for buildings end-uses and transportation modes.
        # Industry has only one value in 'subsector' or 'sector'.
        subsector_col = None
        for c in ("subsector", "sector"):
            if c in df.columns:
                subsector_col = c
                break
        if subsector_col is None:
            raise ValueError(f"Sheet {sheet!r} has neither 'subsector' nor 'sector' column")

        # State / region column
        if "region" not in df.columns:
            raise ValueError(f"Sheet {sheet!r} has no 'region' column")

        # Year columns: 4-digit ints
        year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
        if not year_cols:
            raise ValueError(
                f"Sheet {sheet!r} has no 4-digit year columns. Got: {list(df.columns)}"
            )

        # Melt to long
        # Include 'scenario' column in id_vars if the xlsx contains it,
        # so callers can filter by GCAM scenario (see --gcam-scenario in main).
        id_vars = ["region", subsector_col]
        if "scenario" in df.columns:
            id_vars = ["scenario", "region", subsector_col]
        long = df.melt(
            id_vars=id_vars,
            value_vars=year_cols,
            var_name="year",
            value_name="value",
        )
        if "scenario" not in long.columns:
            long["scenario"] = ""
        long["state"] = long["region"].map(normalize_state)
        long["sector_family"] = family
        long = long.rename(columns={subsector_col: "gcam_subsector"})
        long["gcam_subsector"] = long["gcam_subsector"].astype(str).str.strip()
        long["year"] = long["year"].astype(int)

        # Drop rows that aren't state-level (USA rollup, etc.)
        long = long.dropna(subset=["state"])
        long = long[long["value"].notna()]
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])

        # Convert to MWh using the caller-supplied factor.
        # Default: 2.7777778e8 (EJ → MWh). See --input-unit CLI argument.
        long["mwh"] = long["value"] * unit_to_mwh

        rows.append(long[["scenario", "state", "sector_family", "gcam_subsector", "year", "mwh", "value"]])
        print(f"  Sheet {sheet!r} ({family}): {len(long)} state-year rows, "
              f"{long['gcam_subsector'].nunique()} unique subsectors")

    if not rows:
        raise RuntimeError("No usable rows extracted from xlsx.")
    return pd.concat(rows, ignore_index=True)


def map_to_efs_groups(gcam_long: pd.DataFrame) -> pd.DataFrame:
    """Map GCAM (sector_family, gcam_subsector) -> EFS subsector_group.

    Drops rollup rows that don't appear in GCAM_TO_EFS_GROUP. Hierarchical
    transportation rollups (trn_pass, trn_pass_road, etc.) are intentionally
    NOT in the map and are dropped here.
    """
    keys = list(zip(gcam_long["sector_family"], gcam_long["gcam_subsector"]))
    gcam_long = gcam_long.copy()
    gcam_long["efs_group"] = [GCAM_TO_EFS_GROUP.get(k) for k in keys]

    # Inventory unmapped rows for transparency
    unmapped = gcam_long[gcam_long["efs_group"].isna()]
    if not unmapped.empty:
        unique_unmapped = unmapped.groupby(["sector_family", "gcam_subsector"]).size()
        # Hierarchical transportation rollups are expected to be unmapped
        expected_unmapped = {
            ("transportation", "trn_pass"),
            ("transportation", "trn_pass_road"),
            ("transportation", "trn_pass_road_LDV"),
            ("transportation", "trn_freight"),
        }
        truly_unexpected = [k for k in unique_unmapped.index if k not in expected_unmapped]
        print("\nUnmapped GCAM (sector_family, gcam_subsector) keys:")
        for k, n in unique_unmapped.items():
            tag = "  [expected rollup]" if k in expected_unmapped else "  [UNEXPECTED]"
            print(f"  {k}: {n} rows{tag}")
        if truly_unexpected:
            print(
                f"\nWARNING: {len(truly_unexpected)} unexpected unmapped key(s). "
                f"Add them to GCAM_TO_EFS_GROUP if they should be included."
            )

    mapped = gcam_long.dropna(subset=["efs_group"]).copy()
    return mapped


def build_scaling_inputs(
    mapped: pd.DataFrame,
    scenario: str,
    years: List[int],
) -> pd.DataFrame:
    """Aggregate mapped GCAM rows to scaling_inputs_MWh.csv format.

    The 'subsector_group' column is written as a comma-joined list of the
    actual subsector STRINGS that the unscaled CSV uses, because main.py
    parses this column with `subsector_group.split(',')` and matches each
    piece against the 'subsector' column of the unscaled shape file.

    For example:
      EFS group 'industrial (aggregate)' covers 3 subsectors, so the row
      written here has subsector_group =
        'industrial machine drives, industrial other, industrial process heat'

    Output columns:
        scenario, subsector_group, year, alabama, ..., wyoming
    """
    # Sum across all GCAM end-uses that map to the same EFS group
    agg = (
        mapped[mapped["year"].isin(years)]
        .groupby(["state", "efs_group", "year"], as_index=False)["mwh"]
        .sum()
    )

    # Pivot to one row per (efs_group, year) and one column per state
    rows = []
    all_groups = sorted(EFS_GROUP_TO_SUBSECTORS.keys())
    for group in all_groups:
        # Convert the EFS group label into the comma-joined subsector string
        # main.py expects.
        subsector_string = ", ".join(EFS_GROUP_TO_SUBSECTORS[group])
        for year in sorted(years):
            row = {
                "scenario": scenario,
                "subsector_group": subsector_string,
                "year": int(year),
            }
            sub = agg[(agg["efs_group"] == group) & (agg["year"] == year)]
            by_state = dict(zip(sub["state"], sub["mwh"]))
            for st in UCS_STATE_COLUMNS:
                row[st] = float(by_state.get(st, 0.0))
            rows.append(row)

    out = pd.DataFrame(rows, columns=["scenario", "subsector_group", "year"] + UCS_STATE_COLUMNS)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gcam-xlsx", required=True, help="Path to example_gcam_query.xlsx")
    p.add_argument("--scenario", default="gcam_default", help="Scenario name in output CSV")
    p.add_argument("--gcam-scenario", default=None,
                   help=(
                       "If the xlsx contains a 'scenario' column with multiple GCAM scenarios, "
                       "filter to this one before aggregating. "
                       "If omitted and multiple scenarios are found, raises ValueError."
                   ))
    p.add_argument("--input-unit", choices=["EJ", "TWh", "MWh"], default="EJ",
                   help=(
                       "Energy unit of the GCAM xlsx values. "
                       "EJ (default): 1 EJ = 2.7777778e8 MWh. "
                       "TWh: 1 TWh = 1e6 MWh. "
                       "MWh: values are already in MWh (factor=1.0)."
                   ))
    p.add_argument("--output", default="scaling_inputs_MWh.csv", help="Output CSV path")
    p.add_argument("--years", default="2025,2030,2035,2040,2045,2050",
                   help="Comma-separated list of years to extract")
    args = p.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    unit_to_mwh = {"EJ": 2.7777778e8, "TWh": 1e6, "MWh": 1.0}[args.input_unit]
    print(f"Reading {args.gcam_xlsx} (input unit: {args.input_unit}, "
          f"conversion factor: {unit_to_mwh:.6g} MWh per unit)")
    gcam_long = load_gcam_xlsx(Path(args.gcam_xlsx), unit_to_mwh=unit_to_mwh)
    print(f"\nTotal long-format rows: {len(gcam_long):,}")

    # Filter by GCAM scenario if the xlsx has a scenario column
    gcam_scenarios = gcam_long["scenario"].dropna().astype(str).unique()
    gcam_scenarios = [s for s in gcam_scenarios if s.strip() and s != ""]
    if len(gcam_scenarios) > 1:
        if args.gcam_scenario is None:
            raise ValueError(
                f"xlsx contains multiple GCAM scenarios: {sorted(gcam_scenarios)}.\n"
                f"Pass --gcam-scenario <name> to select one."
            )
        gcam_long = gcam_long[gcam_long["scenario"].astype(str) == args.gcam_scenario].copy()
        print(f"\nFiltered to GCAM scenario: {args.gcam_scenario!r} "
              f"({len(gcam_long):,} rows remaining)")
    elif len(gcam_scenarios) == 1:
        print(f"\nSingle GCAM scenario detected: {gcam_scenarios[0]!r}")

    mapped = map_to_efs_groups(gcam_long)
    print(f"\nMapped rows: {len(mapped):,}")
    print(f"EFS groups present: {mapped['efs_group'].nunique()}")
    for g in sorted(mapped["efs_group"].unique()):
        n = (mapped["efs_group"] == g).sum()
        print(f"  {g}: {n} rows")

    scaling = build_scaling_inputs(mapped, args.scenario, years)
    out_path = Path(args.output)
    scaling.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: shape={scaling.shape}")
    print(f"  scenarios: {sorted(scaling['scenario'].unique())}")
    print(f"  subsector_groups: {sorted(scaling['subsector_group'].unique())}")
    print(f"  years: {sorted(scaling['year'].unique())}")


if __name__ == "__main__":
    main()
