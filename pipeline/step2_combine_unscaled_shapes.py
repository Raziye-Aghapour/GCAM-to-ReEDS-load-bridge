"""
step2_combine_unscaled_shapes.py
=================================

Combine an EFS base unscaled shape CSV with replacement shapes from ResStock,
ComStock, and EVI-Pro Lite. Per-subsector REPLACE semantics: for every
subsector present in a replacement file, the corresponding rows in the EFS
base are dropped and the replacement rows are inserted. Subsectors only
present in the base pass through unchanged.

UCS unscaled-shape CSV schema (all inputs and outputs):
  Columns: sector, subsector, weather_datetime, alabama, alaska, ..., wyoming
  Rows:    8760 per subsector (one full weather year per subsector)
  All values are non-negative; magnitude is meaningless (only shape matters).

Expected subsector strings (must match EFS_GROUP_TO_SUBSECTORS in step 1):
  Residential (4 subsectors):
    - residential space heating and cooling
    - residential water heating
    - residential clothes and dish washing/drying
    - residential other
  Commercial (3 subsectors):
    - commercial space heating and cooling
    - commercial water heating
    - commercial other
  Industrial (3 subsectors — base only, not currently replaced):
    - industrial machine drives
    - industrial other
    - industrial process heat
  Transportation (4 subsectors):
    - transportation light-duty vehicles            <-- replaced by EVI-Pro
    - transportation medium-duty trucks
    - transportation heavy-duty trucks
    - transportation other

Usage:
  python step2_combine_unscaled_shapes.py \
      --efs-base /path/to/efs_unscaled_2018.csv.gz \
      --resstock /path/to/resstock_unscaled.csv.gz \
      --comstock /path/to/comstock_unscaled.csv.gz \
      --evipro /path/to/evipro_tx_unscaled.csv.gz \
      --output combined_unscaled_2018.csv.gz

Any of --resstock, --comstock, --evipro can be omitted; the EFS base shape
for those subsectors will pass through unchanged.

Validation:
- All shape files must have identical 'weather_datetime' axes (same 8760 hours).
- Replacement files must have at least one subsector that overlaps with EFS.
- Replacement files cannot introduce new subsectors not present in EFS.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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

REQUIRED_COLS = ["sector", "subsector", "weather_datetime"] + UCS_STATE_COLUMNS

# Subsectors each replacement source is allowed to provide
RESSTOCK_SUBSECTORS = {
    "residential space heating and cooling",
    "residential water heating",
    "residential clothes and dish washing/drying",
    "residential other",
}
COMSTOCK_SUBSECTORS = {
    "commercial space heating and cooling",
    "commercial water heating",
    "commercial other",
}
EVIPRO_SUBSECTORS = {
    "transportation light-duty vehicles",
}


def load_ucs_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a UCS unscaled-shape CSV and validate the schema."""
    if not path.exists():
        raise FileNotFoundError(f"{label}: file not found: {path}")
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    df = pd.read_csv(path, compression=compression)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required columns {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    df = df[REQUIRED_COLS].copy()
    df["weather_datetime"] = pd.to_datetime(df["weather_datetime"])
    return df


def validate_8760_per_subsector(df: pd.DataFrame, label: str) -> None:
    """Ensure each subsector has exactly 8760 rows AND 8760 unique hourly timestamps.

    Checking only nunique() catches duplicated timestamps but misses an extra
    row that happens to be a new unique timestamp (e.g. a stray 8761st hour).
    Checking only count() catches extra rows but misses duplicate timestamps.
    Both checks together are necessary and sufficient.
    """
    stats = df.groupby("subsector")["weather_datetime"].agg(
        rows="count", unique="nunique"
    )
    bad = stats[(stats["rows"] != 8760) | (stats["unique"] != 8760)]
    if not bad.empty:
        raise ValueError(
            f"{label}: subsectors must have exactly 8760 rows "
            f"and 8760 unique timestamps.\n"
            f"{bad.to_string()}"
        )


def validate_aligned_timestamps(
    base: pd.DataFrame,
    repl: pd.DataFrame,
    repl_label: str,
) -> None:
    """Both files must have the same 8760 hourly timestamps for each subsector."""
    base_ts = set(base["weather_datetime"].unique())
    repl_ts = set(repl["weather_datetime"].unique())
    if base_ts != repl_ts:
        only_in_base = sorted(base_ts - repl_ts)[:3]
        only_in_repl = sorted(repl_ts - base_ts)[:3]
        raise ValueError(
            f"{repl_label}: weather_datetime axis does not match base.\n"
            f"  In base only (first 3): {only_in_base}\n"
            f"  In {repl_label} only (first 3): {only_in_repl}\n"
            f"  Base has {len(base_ts)} unique hours; {repl_label} has {len(repl_ts)}.\n"
            f"Both must use the same weather year and hour-of-year alignment."
        )


def validate_allowed_subsectors(
    repl: pd.DataFrame,
    allowed: Iterable[str],
    repl_label: str,
) -> List[str]:
    """Replacement file's subsectors must be a non-empty subset of `allowed`."""
    have = set(repl["subsector"].unique())
    extra = have - set(allowed)
    if extra:
        raise ValueError(
            f"{repl_label}: contains subsectors not allowed for this source: "
            f"{sorted(extra)}.\nAllowed subsectors: {sorted(allowed)}"
        )
    if not have:
        raise ValueError(f"{repl_label}: no subsectors found")
    print(f"  {repl_label} replaces {len(have)} subsector(s): {sorted(have)}")
    return sorted(have)


def replace_subsectors(
    base: pd.DataFrame,
    repl: pd.DataFrame,
    repl_label: str,
) -> pd.DataFrame:
    """Drop rows in base whose subsector matches any in repl, then concat repl."""
    repl_subs = set(repl["subsector"].unique())
    not_in_base = repl_subs - set(base["subsector"].unique())
    if not_in_base:
        raise ValueError(
            f"{repl_label}: subsectors not present in EFS base, cannot replace: "
            f"{sorted(not_in_base)}.\n"
            f"Either rename these subsectors to match EFS strings, or add them "
            f"to the EFS base file first."
        )
    base_keep = base[~base["subsector"].isin(repl_subs)].copy()
    return pd.concat([base_keep, repl], ignore_index=True)


def combine(
    efs_base: Path,
    resstock: Optional[Path],
    comstock: Optional[Path],
    evipro: Optional[Path],
) -> pd.DataFrame:
    print(f"Loading EFS base from {efs_base}")
    base = load_ucs_csv(efs_base, "EFS base")
    validate_8760_per_subsector(base, "EFS base")
    base_subs = sorted(base["subsector"].unique())
    print(f"  EFS base has {len(base_subs)} subsectors")
    for s in base_subs:
        print(f"    - {s}")

    if resstock:
        print(f"\nLoading ResStock from {resstock}")
        repl = load_ucs_csv(resstock, "ResStock")
        validate_8760_per_subsector(repl, "ResStock")
        validate_aligned_timestamps(base, repl, "ResStock")
        validate_allowed_subsectors(repl, RESSTOCK_SUBSECTORS, "ResStock")
        base = replace_subsectors(base, repl, "ResStock")

    if comstock:
        print(f"\nLoading ComStock from {comstock}")
        repl = load_ucs_csv(comstock, "ComStock")
        validate_8760_per_subsector(repl, "ComStock")
        validate_aligned_timestamps(base, repl, "ComStock")
        validate_allowed_subsectors(repl, COMSTOCK_SUBSECTORS, "ComStock")
        base = replace_subsectors(base, repl, "ComStock")

    if evipro:
        print(f"\nLoading EVI-Pro from {evipro}")
        repl = load_ucs_csv(evipro, "EVI-Pro")
        validate_8760_per_subsector(repl, "EVI-Pro")
        validate_aligned_timestamps(base, repl, "EVI-Pro")
        validate_allowed_subsectors(repl, EVIPRO_SUBSECTORS, "EVI-Pro")
        base = replace_subsectors(base, repl, "EVI-Pro")

    # Final validation
    validate_8760_per_subsector(base, "combined")
    base = base.sort_values(["sector", "subsector", "weather_datetime"]).reset_index(drop=True)
    return base


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--efs-base", required=True,
                   help="Path to EFS unscaled CSV (e.g. eer_load_shapes baseline)")
    p.add_argument("--resstock", help="Optional ResStock unscaled CSV")
    p.add_argument("--comstock", help="Optional ComStock unscaled CSV")
    p.add_argument("--evipro", help="Optional EVI-Pro unscaled CSV")
    p.add_argument("--output", required=True, help="Output combined unscaled CSV path")
    args = p.parse_args()

    combined = combine(
        efs_base=Path(args.efs_base),
        resstock=Path(args.resstock) if args.resstock else None,
        comstock=Path(args.comstock) if args.comstock else None,
        evipro=Path(args.evipro) if args.evipro else None,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if out.suffix.lower() == ".gz" else None
    combined.to_csv(out, index=False, compression=compression)

    print(f"\nWrote {out}")
    print(f"  Total rows: {len(combined):,} "
          f"({combined['subsector'].nunique()} subsectors × 8760 hours)")
    print(f"  Subsectors: {sorted(combined['subsector'].unique())}")


if __name__ == "__main__":
    main()
