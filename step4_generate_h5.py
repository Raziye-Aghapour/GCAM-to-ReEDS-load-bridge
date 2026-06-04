"""
step4_generate_h5.py
=====================

Read scaled shape CSVs (from step 3) and produce a ReEDS-loadable
*_load_hourly.h5 file that drops directly into ReEDS-2.0/inputs/load/.

Replaces the role of generate_scenarios.py from the eer_load_shapes repo,
but works with our 11-subsector_group scaling_inputs structure (built from
example_gcam_query.xlsx) rather than the data-center-only original.

Pipeline orientation
--------------------
  step 1: GCAM xlsx       -> scaling_inputs_MWh.csv
  step 2: 4 source shapes -> combined_unscaled_{year}.csv.gz  (once per year)
  step 3: combined+scaling -> scaled_shapes/<scenario>/<year>.csv.gz
  step 4: scaled_shapes   -> <scenario>_load_hourly.h5     <-- this script

What it does
------------
For a given scenario it reads each year's scaled CSV, sums across subsectors
to get a (weather_datetime) x state hourly demand frame, then disaggregates
each state's load to ReEDS BAs using load_factors.csv shares. The H5 schema
matches generate_scenarios.write_profile_to_h5 byte-for-byte:

  data        (n_years*8760, n_BA) float64 gzip
  index_0     int64 (year)
  index_1     S30   (weather_datetime ISO strings)
  index_names ['year', 'weather_datetime'] as bytes
  columns     BA names as bytes

Optional features:
  --historical-years  zero-pads years 2010-2024 (or whatever you give it)
                      so the H5 covers both history and projection
  --template-h5       merge mode: read an existing template H5, ADD or
                      REPLACE selected state-BAs and years with the scaled
                      data. Use this to layer GCAM-driven shapes on top of
                      an existing ReEDS baseline.

Usage:
  # From-scratch single-scenario H5
  python step4_generate_h5.py \
      --scaled-dir scaled_shapes \
      --scenario gcam_default \
      --load-factors load_factors.csv \
      --output gcam_default_load_hourly.h5 \
      --historical-years 2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024

  # Merge into existing template
  python step4_generate_h5.py \
      --scaled-dir scaled_shapes \
      --scenario gcam_default \
      --load-factors load_factors.csv \
      --template-h5 ReEDS-2.0/inputs/load/EER_Baseline_AEO2023_load_hourly.h5 \
      --operation replace \
      --output ReEDS-2.0/inputs/load/EER_GCAMdefault_load_hourly.h5
"""
from __future__ import annotations
import argparse
import datetime
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
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


# ---------------------------------------------------------------------------
# H5 read / write (byte-compatible with generate_scenarios.write_profile_to_h5)
# ---------------------------------------------------------------------------

def write_reeds_h5(df: pd.DataFrame, path: Path, compression_opts: int = 4) -> Path:
    """Write a (year, weather_datetime) x BA DataFrame to ReEDS H5 format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.index.nlevels != 2:
        raise ValueError(f"Expected 2-level MultiIndex; got {df.index.nlevels}")
    if list(df.index.names) != ["year", "weather_datetime"]:
        raise ValueError(
            f"Index names must be ['year','weather_datetime']; got {list(df.index.names)}"
        )

    df = df.astype(np.float64)

    with h5py.File(path, "w") as f:
        years = df.index.get_level_values(0).to_numpy()
        f.create_dataset("index_0", data=years.astype(np.int64))

        dts = df.index.get_level_values(1)
        if pd.api.types.is_datetime64_any_dtype(dts):
            iso = pd.to_datetime(dts).to_series().apply(datetime.datetime.isoformat)
        else:
            iso = pd.to_datetime(dts.astype(str)).to_series().apply(datetime.datetime.isoformat)
        iso_arr = np.array(iso.reset_index(drop=True).tolist(), dtype="S30")
        f.create_dataset("index_1", data=iso_arr, dtype="S30")

        idx_names = list(df.index.names)
        max_n = max(len(n) for n in idx_names)
        f.create_dataset("index_names", data=np.array(idx_names, dtype=f"S{max_n}"),
                         dtype=f"S{max_n}")

        col_strs = [str(c) for c in df.columns]
        max_c = max(max(len(c) for c in col_strs), 4)
        f.create_dataset("columns", data=np.array(col_strs, dtype=f"S{max_c}"),
                         dtype=f"S{max_c}")

        f.create_dataset(
            "data",
            data=df.values,
            dtype=np.float64,
            compression="gzip",
            compression_opts=compression_opts,
        )
    return path


def read_reeds_h5(path: Path) -> pd.DataFrame:
    """Read a ReEDS-format H5 (matches generate_scenarios.read_h5py_file)."""
    path = Path(path)
    valid_data_keys = {"data", "cf", "load", "evload"}
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        datakey = list(set(keys) & valid_data_keys)
        if len(datakey) > 1:
            raise ValueError(f"Multiple data keys in {path}: {datakey}")
        datakey = datakey[0] if datakey else None
        df = pd.DataFrame(f[datakey][:]) if datakey else pd.DataFrame()

        if "columns" in keys:
            df.columns = (
                pd.Series(f["columns"][:])
                .map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
                .values
            )

        idx_cols = sorted([c for c in keys if re.match(r"index_[0-9]", c)])
        if idx_cols:
            for c in idx_cols:
                vals = f[c][:]
                if vals.dtype.kind == "S":
                    vals = pd.Series(vals).str.decode("utf-8").values
                df[c] = vals
            df = df.set_index(idx_cols)

        if "index_names" in keys:
            names = (
                pd.Series(f["index_names"][:])
                .map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
                .values
            )
            df.index.names = names
    return df


# ---------------------------------------------------------------------------
# Load factors and BA disaggregation
# ---------------------------------------------------------------------------

def load_ba_factors(path: Path) -> pd.DataFrame:
    """Load a BA-share CSV. Accepts column variants:
        ba/reeds_ba/r/region, st/abbr/state_abbr, state/state_name,
        load_factor/factor/share/fraction.
    Returns DataFrame with columns ['ba','state','share'].
    """
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None
    ba_col = pick("ba", "reeds_ba", "r", "region")
    share_col = pick("load_factor", "factor", "share", "fraction", "load_share")
    state_full_col = pick("state", "state_name")
    state_abbr_col = pick("st", "abbr", "state_abbr")
    if ba_col is None or share_col is None:
        raise ValueError(
            f"Missing BA/share columns in {path}. Got: {list(df.columns)}"
        )
    out = pd.DataFrame()
    out["ba"] = df[ba_col].astype(str).str.strip()
    out["share"] = pd.to_numeric(df[share_col], errors="coerce")
    if state_full_col:
        raw = df[state_full_col].astype(str).str.strip()
        out["state"] = raw.str.lower()
        # FIX [Fix5a]: if values look like 2-letter abbreviations (TX, CA …),
        # remap them even though the column is named 'state' not 'st'.
        unresolved = ~out["state"].isin(UCS_STATE_COLUMNS)
        if unresolved.any():
            remapped = raw.str.upper().map(US_ABBR_TO_NAME)
            out.loc[unresolved, "state"] = remapped[unresolved]
    elif state_abbr_col:
        out["state"] = (
            df[state_abbr_col].astype(str).str.strip().str.upper().map(US_ABBR_TO_NAME)
        )
    else:
        raise ValueError(f"{path}: need 'state' or 'st' column")
    if out["state"].isna().any():
        bad = df[out["state"].isna()].head()
        raise ValueError(f"Could not resolve states for:\n{bad}")
    if out["share"].isna().any():
        bad = df[out["share"].isna()].head()
        raise ValueError(f"Non-numeric BA shares:\n{bad}")
    return out


def normalize_state_shares(ba_factors: pd.DataFrame, states: Iterable[str]) -> pd.DataFrame:
    out = ba_factors.copy()
    out["share_normalized"] = out["share"]
    for st in states:
        m = out["state"] == st
        if not m.any():
            continue
        s = out.loc[m, "share"].sum()
        if s <= 0:
            raise ValueError(f"BA shares for '{st}' sum to non-positive: {s}")
        out.loc[m, "share_normalized"] = out.loc[m, "share"] / s
    return out


def disaggregate_states_to_bas(
    state_year_df: pd.DataFrame,
    ba_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Convert (year, weather_datetime) x state -> (year, weather_datetime) x BA.

    Every BA in `ba_factors` becomes a column (zero-fill for states with no
    data). Energy conservation is verified to within 1e-8 relative.
    """
    state_year_df = state_year_df.copy()
    all_states_in_factors = set(ba_factors["state"].unique())
    for st in all_states_in_factors:
        if st not in state_year_df.columns:
            state_year_df[st] = 0.0

    states_to_normalize = [
        s for s in all_states_in_factors
        if s in state_year_df.columns and state_year_df[s].sum() > 0
    ]
    factors = normalize_state_shares(ba_factors, states_to_normalize)

    ba_columns: Dict[str, np.ndarray] = {}
    for _, row in factors.iterrows():
        ba = str(row["ba"])
        st = str(row["state"])
        share = float(row.get("share_normalized", row["share"]))
        # FIX [Fix5b]: accumulate with += so a BA that receives shares from
        # multiple states (or appears in multiple load_factors rows) is handled
        # correctly. dict overwrite would silently lose earlier contributions.
        vals = (np.zeros(len(state_year_df), dtype=np.float64)
                if state_year_df[st].sum() == 0
                else state_year_df[st].to_numpy(dtype=np.float64) * share)
        if ba in ba_columns:
            ba_columns[ba] += vals
        else:
            ba_columns[ba] = vals

    ba_df = pd.DataFrame(ba_columns, index=state_year_df.index)

    in_total = state_year_df[
        [s for s in states_to_normalize if s in state_year_df.columns]
    ].sum().sum()
    out_total = ba_df.sum().sum()
    if in_total > 0:
        rel = abs(out_total - in_total) / in_total
        if rel > 1e-8:
            raise RuntimeError(
                f"Energy mismatch after disaggregation: state={in_total:.3f}, "
                f"BA={out_total:.3f}, rel={rel:.2e}"
            )

    return ba_df


# ---------------------------------------------------------------------------
# Reading scaled CSVs and assembling per-year state hourly frames
# ---------------------------------------------------------------------------

def load_scaled_csv(path: Path) -> pd.DataFrame:
    """Load a scaled-shape CSV produced by step 3."""
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    df = pd.read_csv(path, compression=compression)
    df.columns = [str(c).strip() for c in df.columns]
    needed = ["sector", "subsector", "weather_datetime"] + UCS_STATE_COLUMNS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing[:5]}")
    df["weather_datetime"] = pd.to_datetime(df["weather_datetime"])
    return df[needed].copy()


def _assert_efs_calendar(idx: pd.DatetimeIndex, label: str = "") -> None:
    """Verify the EFS 2012 leap-year calendar convention.

    Convention (from EVI-Pro notebook, verified):
      WEATHER_YEAR = 2012 (leap year, 366 days = 8784 hours)
      Keep Feb 29, drop Dec 31  →  8760 hours: Jan 1 00:00 → Dec 30 23:00
    """
    if len(idx) != 8760:
        raise ValueError(
            f"{label}: expected 8760 unique hours; got {len(idx)}.\n"
            "EFS convention: WEATHER_YEAR=2012, keep Feb 29, drop Dec 31."
        )
    dts = pd.to_datetime(idx)
    has_feb29 = ((dts.month == 2) & (dts.day == 29)).any()
    has_dec31 = ((dts.month == 12) & (dts.day == 31)).any()
    if not has_feb29:
        raise ValueError(
            f"{label}: Feb 29 is absent. "
            "EFS HOURLY_AXIS uses WEATHER_YEAR=2012 (leap year): "
            "Feb 29 must be present and Dec 31 must be absent. "
            "A source missing Feb 29 causes a 24-hour misalignment for all "
            "hours after position 1368 (Mar 1 onward)."
        )
    if has_dec31:
        raise ValueError(
            f"{label}: Dec 31 is present. "
            "EFS convention drops Dec 31 (axis ends Dec 30 23:00). "
            "Check that WEATHER_YEAR=2012 and the drop targets Dec 31 not Feb 29."
        )


def aggregate_subsectors_to_state_hourly(scaled: pd.DataFrame) -> pd.DataFrame:
    """Sum scaled shape across subsectors -> (weather_datetime) x state.

    FIX [Fix3b]: validates the EFS 2012 calendar convention
    (keep Feb 29, drop Dec 31) on the weather_datetime index.
    """
    state_cols = [c for c in UCS_STATE_COLUMNS if c in scaled.columns]
    out = scaled.groupby("weather_datetime")[state_cols].sum().sort_index()
    _assert_efs_calendar(out.index, label="scaled shape")
    return out


def build_year_indexed_states(
    scaled_dir: Path,
    scenario: str,
    years: List[int],
) -> pd.DataFrame:
    """Stack per-year scaled CSVs into a (year, weather_datetime) x state frame."""
    pieces = []
    for y in sorted(years):
        path = scaled_dir / scenario / f"{int(y)}.csv.gz"
        if not path.exists():
            path_csv = scaled_dir / scenario / f"{int(y)}.csv"
            if path_csv.exists():
                path = path_csv
            else:
                raise FileNotFoundError(f"No scaled CSV for {scenario}/{y}: {path}")
        scaled = load_scaled_csv(path)
        state_hourly = aggregate_subsectors_to_state_hourly(scaled)
        state_hourly.index.name = "weather_datetime"
        state_hourly = state_hourly.reset_index()
        state_hourly["year"] = int(y)
        state_hourly = state_hourly.set_index(["year", "weather_datetime"])
        pieces.append(state_hourly)
        print(f"  Loaded {path.name}: total={state_hourly.values.sum():,.1f} MWh")
    return pd.concat(pieces).sort_index()


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------

def build_h5_from_scratch(
    scaled_dir: Path,
    scenario: str,
    years: List[int],
    load_factors_csv: Path,
    output_h5: Path,
    historical_years: Optional[List[int]] = None,
) -> Path:
    print(f"Building H5 from scratch for scenario={scenario!r}, years={years}")
    state_year_df = build_year_indexed_states(scaled_dir, scenario, years)
    ba_factors = load_ba_factors(load_factors_csv)
    ba_df = disaggregate_states_to_bas(state_year_df, ba_factors)

    if historical_years:
        first_year_dts = ba_df.index.get_level_values(1).unique()
        zero_frames = []
        for y in sorted(historical_years):
            zframe = pd.DataFrame(
                np.zeros((8760, ba_df.shape[1]), dtype=np.float64),
                index=pd.MultiIndex.from_product(
                    [[int(y)], first_year_dts],
                    names=["year", "weather_datetime"],
                ),
                columns=ba_df.columns,
            )
            zero_frames.append(zframe)
        ba_df = pd.concat(zero_frames + [ba_df]).sort_index()

    write_reeds_h5(ba_df, output_h5)
    return output_h5


def merge_into_template(
    template_h5: Path,
    scaled_dir: Path,
    scenario: str,
    years: List[int],
    load_factors_csv: Path,
    output_h5: Path,
    states_to_modify: Optional[List[str]],
    operation: str = "replace",
) -> Path:
    """Layer scaled-shape data onto an existing template H5.

    operation='replace': overwrite template values in selected state-BAs/years.
    operation='add'    : add scaled values on top of template values.
    states_to_modify   : which states' BAs to modify; default = all states with
                         nonzero data in the scaled output.
    """
    if operation not in ("add", "replace"):
        raise ValueError(f"operation must be 'add' or 'replace'; got {operation}")

    template = read_reeds_h5(template_h5)
    if list(template.index.names) != ["year", "weather_datetime"]:
        raise ValueError(
            f"Template index names must be ['year','weather_datetime']; "
            f"got {list(template.index.names)}"
        )

    state_year_df = build_year_indexed_states(scaled_dir, scenario, years)
    ba_factors = load_ba_factors(load_factors_csv)
    if states_to_modify is None:
        # default: all states with nonzero data
        sums = state_year_df.sum()
        states_to_modify = [s for s in sums.index if sums[s] > 0]
    print(f"  Modifying {len(states_to_modify)} states' BAs in template")

    # Restrict ba_factors to target states only (we don't touch other BAs)
    ba_factors_subset = ba_factors[ba_factors["state"].isin(states_to_modify)].copy()
    mod_ba = disaggregate_states_to_bas(state_year_df, ba_factors_subset)

    missing = [c for c in mod_ba.columns if c not in template.columns]
    if missing:
        raise ValueError(
            f"Template missing target BAs: {missing}.\n"
            f"Either the load_factors.csv has BAs not in this template, or "
            f"the template uses a different BA naming convention."
        )

    # Align timestamps by hour-of-year (template may use 2012, source 2018)
    template_dts = template.index.get_level_values(1).unique()
    source_dts = mod_ba.index.get_level_values(1).unique()
    # EFS/pipeline calendar convention (from EVI-Pro notebook, verified):
    #   WEATHER_YEAR = 2012 (leap year, 366 days = 8784 hours)
    #   Keep Feb 29, drop Dec 31  →  8760 hours: Jan 1 00:00 → Dec 30 23:00
    # Validate both sides have exactly 8760 unique hours.
    if len(source_dts) != 8760:
        raise ValueError(
            f"Source has {len(source_dts)} unique datetimes; expected 8760.\n"
            "EFS convention: WEATHER_YEAR=2012, keep Feb 29, drop Dec 31 "
            "(Jan 1 00:00 → Dec 30 23:00)."
        )
    if len(template_dts) != 8760:
        raise ValueError(
            f"Template has {len(template_dts)} unique datetimes; expected 8760."
        )
    # Check source (not template) for Feb 29 — we only control the source side.
    # If Feb 29 is absent from source, hours 1368+ will be misaligned by 24 hours.
    _source_has_feb29 = any(
        "02-29" in str(dt) or
        (hasattr(dt, "month") and getattr(dt, "month") == 2 and getattr(dt, "day") == 29)
        for dt in source_dts
    )
    if not _source_has_feb29:
        raise ValueError(
            "Source does not contain Feb 29 timestamps.\n"
            "EFS HOURLY_AXIS uses WEATHER_YEAR=2012 (leap year): Feb 29 is retained\n"
            "and Dec 31 is dropped. A source missing Feb 29 will cause a 24-hour\n"
            "misalignment for all hours after position 1368 (Mar 1 onward)."
        )
    dt_map = dict(zip(sorted(source_dts), sorted(template_dts)))
    new_index = pd.MultiIndex.from_tuples(
        [(y, dt_map[dt]) for (y, dt) in mod_ba.index],
        names=["year", "weather_datetime"],
    )
    mod_ba.index = new_index

    out = template.copy()
    n_modified = 0
    for year in sorted(set(mod_ba.index.get_level_values(0))):
        if year not in out.index.get_level_values(0):
            print(f"  WARN: template missing year={year}; skipping")
            continue
        tmpl_idx = out.index[out.index.get_level_values(0) == year]
        mod_idx = mod_ba.index[mod_ba.index.get_level_values(0) == year]
        for ba in mod_ba.columns:
            mod_vals = mod_ba.loc[mod_idx, ba].to_numpy(dtype=np.float64)
            if operation == "add":
                out.loc[tmpl_idx, ba] = (
                    out.loc[tmpl_idx, ba].to_numpy(dtype=np.float64) + mod_vals
                )
            else:
                out.loc[tmpl_idx, ba] = mod_vals
            n_modified += 1
    print(f"  Modified {n_modified} BA-year columns")

    write_reeds_h5(out, output_h5)
    return output_h5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scaled-dir", required=True, help="Directory of step-3 output")
    p.add_argument("--scenario", required=True, help="Scenario name (subdir of --scaled-dir)")
    p.add_argument("--load-factors", required=True, help="BA share CSV")
    p.add_argument("--output", required=True, help="Output H5 path")
    p.add_argument("--years", help="Comma-separated years (default: all CSVs in scenario dir)")
    p.add_argument("--historical-years", help="Comma-separated historical years to zero-pad (from-scratch only)")
    p.add_argument("--template-h5", help="Template H5 to merge into (turns on merge mode)")
    p.add_argument("--operation", choices=["add", "replace"], default="replace",
                   help="Merge operation: 'replace' overwrites, 'add' layers on top")
    p.add_argument("--states", help="Comma-separated states to modify in merge mode (default: all nonzero)")
    args = p.parse_args()

    scaled_dir = Path(args.scaled_dir)
    sc_dir = scaled_dir / args.scenario
    if not sc_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {sc_dir}")

    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    else:
        # FIX [Claim3]: p.stem of "2025.csv.gz" is "2025.csv" not "2025" — isdigit() fails.
        # Parse the first dot-delimited token instead.
        years = sorted(
            int(p.name.split(".")[0])
            for p in sc_dir.glob("*.csv*")
            if p.name.split(".")[0].isdigit()
        )
        if not years:
            raise FileNotFoundError(f"No <year>.csv* files in {sc_dir}")
    print(f"Processing years: {years}")

    if args.template_h5:
        states = (
            [s.strip().lower() for s in args.states.split(",")] if args.states else None
        )
        merge_into_template(
            template_h5=Path(args.template_h5),
            scaled_dir=scaled_dir,
            scenario=args.scenario,
            years=years,
            load_factors_csv=Path(args.load_factors),
            output_h5=Path(args.output),
            states_to_modify=states,
            operation=args.operation,
        )
    else:
        hist = None
        if args.historical_years:
            hist = [int(x.strip()) for x in args.historical_years.split(",") if x.strip()]
        build_h5_from_scratch(
            scaled_dir=scaled_dir,
            scenario=args.scenario,
            years=years,
            load_factors_csv=Path(args.load_factors),
            output_h5=Path(args.output),
            historical_years=hist,
        )

    df = read_reeds_h5(Path(args.output))
    print(f"\n{'='*60}\nWrote {args.output}")
    print(f"  Shape: {df.shape}")
    print(f"  Years: {sorted(df.index.get_level_values(0).unique())[:3]}..."
          f"{sorted(df.index.get_level_values(0).unique())[-3:]}")
    print(f"  BAs:   {df.shape[1]}")
    print(f"  Total annual MWh: {df.sum().sum():,.1f}")


if __name__ == "__main__":
    main()
