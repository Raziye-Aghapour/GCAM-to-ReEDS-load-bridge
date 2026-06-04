"""
main.py  (HARMONIZED — replaces the original)
==============================================

CHANGES FROM ORIGINAL:
  1. scale_profile(): removed .astype(int) → values stay float64.
     This makes energy conservation exact and matches step3_scale_shapes.py.
  2. scale_profile() zero-to-positive: removed int() cast → float.
  3. interpolate_scaling_factors(): replaced positional columns[3:] slice with
     explicit UCS_STATE_COLUMNS list, matching every other script in the pipeline.
  4. Added UCS_STATE_COLUMNS constant (same list as step1–step4).
  5. scale_profile() state_columns: now uses UCS_STATE_COLUMNS intersection
     with df.columns, instead of a generic exclude-set approach.
  6. generate_summary_file(): no functional changes.
  7. main(): removed pdb import (debug artifact).

NOTE: main.py is a standalone alternative to step3_scale_shapes.py.
      run_all.py calls step3, not main.py. Use ONE of them, not both.
      Prefer step3 for the full pipeline (better validation, v2 per-year layout).
      Use main.py only if you need the summary_shapes.csv side-output.
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Canonical 51-entry state list — identical to step1/step2/step3/step4
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


def interpolate_scaling_factors(scaling_inputs, scenario, subsector_group, target_year, available_years):
    """
    Return per-state target MWh for (scenario, subsector_group, target_year).
    Interpolates linearly between GCAM snapshot years when target_year falls between them.
    Uses explicit UCS_STATE_COLUMNS rather than positional slice.
    """
    scenario_data = scaling_inputs[
        (scaling_inputs['scenario'] == scenario) &
        (scaling_inputs['subsector_group'] == subsector_group)
    ]

    if scenario_data.empty:
        print(f"Warning: No scaling data for scenario='{scenario}', group='{subsector_group}'")
        return {state: 0.0 for state in UCS_STATE_COLUMNS}

    input_years = sorted(scenario_data['year'].unique())

    if target_year in input_years:
        row = scenario_data[scenario_data['year'] == target_year].iloc[0]
        # FIX [C1]: use explicit UCS_STATE_COLUMNS, not positional columns[3:]
        return {st: float(row[st]) for st in UCS_STATE_COLUMNS if st in row.index}

    if target_year < min(input_years):
        row = scenario_data[scenario_data['year'] == min(input_years)].iloc[0]
        print(f"Warning: {target_year} before first scaling year; using {min(input_years)}")
        return {st: float(row[st]) for st in UCS_STATE_COLUMNS if st in row.index}

    if target_year > max(input_years):
        row = scenario_data[scenario_data['year'] == max(input_years)].iloc[0]
        print(f"Warning: {target_year} after last scaling year; using {max(input_years)}")
        return {st: float(row[st]) for st in UCS_STATE_COLUMNS if st in row.index}

    lower_year = max(y for y in input_years if y < target_year)
    upper_year = min(y for y in input_years if y > target_year)
    lo = scenario_data[scenario_data['year'] == lower_year].iloc[0]
    hi = scenario_data[scenario_data['year'] == upper_year].iloc[0]
    proportion = (target_year - lower_year) / (upper_year - lower_year)

    return {
        st: float(lo[st]) + proportion * (float(hi[st]) - float(lo[st]))
        for st in UCS_STATE_COLUMNS if st in lo.index
    }


def scale_profile(df, scaling_factors, subsector_group):
    """
    Scale 8760 hourly timeseries so annual sum matches scaling_factors (target MWh).

    FIX [B1, B2]: values stay float64 throughout — no .astype(int) truncation.
    This matches step3_scale_shapes.py and preserves energy conservation to float precision.
    """
    scaled_df = df.copy()
    subsectors = [s.strip() for s in subsector_group.split(',')]
    mask = scaled_df['subsector'].isin(subsectors)

    # FIX: use explicit UCS_STATE_COLUMNS intersection instead of exclude-set
    state_columns = [col for col in UCS_STATE_COLUMNS if col in scaled_df.columns]

    for state in state_columns:
        if state not in scaling_factors:
            continue
        target = float(scaling_factors[state])
        group_sum = float(scaled_df.loc[mask, state].sum())

        if group_sum == 0 and target > 0:
            # Zero-to-positive: spread uniformly — keep float, not int
            num_rows = int(mask.sum())
            if num_rows > 0:
                # FIX [B2]: was int(target / num_rows); now float
                scaled_df.loc[mask, state] = target / num_rows
                print(f"  Zero-to-positive scaling: {state}, group='{subsector_group}'")
        elif group_sum > 0:
            # FIX [B1]: was (... * ratio).astype(int); now float
            ratio = target / group_sum
            scaled_df.loc[mask, state] = scaled_df.loc[mask, state] * ratio
        # else group_sum == 0 and target == 0: leave as-is (already zero)

    return scaled_df


def create_original_energy_summary(unscaled_directory, scaling_inputs, output_dir):
    """Create original_energy_values.csv before scaling (diagnostic)."""
    print("Generating original energy summary file...")
    result_data = []
    scenarios = scaling_inputs['scenario'].unique()

    for scenario in scenarios:
        print(f"  Processing scenario: {scenario}")
        subsector_groups = scaling_inputs[scaling_inputs['scenario'] == scenario]['subsector_group'].unique()
        scenario_directory = Path(unscaled_directory) / scenario

        for year_file in os.listdir(scenario_directory):
            if year_file == 'summary_shapes.csv' or not year_file.endswith('.csv.gz'):
                continue
            year = int(year_file.split('.')[0])
            print(f"    Processing year: {year}")
            df = pd.read_csv(os.path.join(scenario_directory, year_file), compression='gzip')
            state_columns = [col for col in UCS_STATE_COLUMNS if col in df.columns]

            for subsector_group in subsector_groups:
                subsectors = [s.strip() for s in subsector_group.split(',')]
                mask = df['subsector'].isin(subsectors)
                row_data = {'scenario': scenario, 'subsector_group': subsector_group, 'year': year}
                for state in state_columns:
                    row_data[state] = float(df.loc[mask, state].sum()) if mask.any() else 0.0
                result_data.append(row_data)

    result_df = pd.DataFrame(result_data)
    output_file = Path(output_dir) / 'original_energy_values.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Original energy summary saved to {output_file}")


def generate_summary_file(scenario, scenario_data, output_dir):
    """Generate summary_shapes.csv (annual totals by state × sector × year)."""
    print(f"  Generating summary_shapes.csv for scenario: {scenario}")
    summary_df = pd.concat(scenario_data.values(), keys=scenario_data.keys(), names=['year'])
    summary_df = summary_df.reset_index('year')
    del summary_df['subsector']
    summary_df = summary_df.groupby(['weather_datetime', 'year', 'sector']).sum()
    summary_df.columns.name = 'state'
    summary_df = summary_df.stack()
    summary_df = summary_df.unstack('year')
    summary_df = summary_df.reorder_levels(['weather_datetime', 'state', 'sector'])
    summary_df = summary_df.sort_index()
    summary_df.columns = [str(int(col)) for col in summary_df.columns]
    summary_df.to_csv(output_dir / 'summary_shapes.csv', index=True)
    print(f"  Created summary_shapes.csv with {len(summary_df)} rows")


def main(args):
    scaled_dir = Path(args.output_dir)
    scaled_dir.mkdir(exist_ok=True)

    scaling_inputs = pd.read_csv(args.scaling_inputs)
    summary_data = {}

    unscaled_directory = Path(args.input_dir)
    scenarios = [d for d in os.listdir(unscaled_directory) if os.path.isdir(unscaled_directory / d)]

    for scenario in scenarios:
        print(f"Processing scenario: {scenario}")
        scenario_directory = unscaled_directory / scenario

        available_years = [
            int(y.split('.')[0]) for y in os.listdir(scenario_directory)
            if y.endswith('.csv.gz') and y != 'summary_shapes.csv'
        ]

        scaled_scenario_dir = scaled_dir / scenario
        scaled_scenario_dir.mkdir(exist_ok=True)
        scenario_data = {}

        for year_file in os.listdir(scenario_directory):
            if year_file in ('summary_shapes.csv',) or not year_file.endswith('.csv.gz'):
                continue

            year = int(year_file.split('.')[0])
            print(f"  Processing year: {year}")
            df = pd.read_csv(os.path.join(scenario_directory, year_file), compression='gzip')
            subsector_groups = scaling_inputs[scaling_inputs['scenario'] == scenario]['subsector_group'].unique()
            scaled_df = df.copy()

            for subsector_group in subsector_groups:
                print(f"    Scaling subsector group: {subsector_group}")
                scaling_factors = interpolate_scaling_factors(
                    scaling_inputs, scenario, subsector_group, year, available_years
                )
                scaled_df = scale_profile(scaled_df, scaling_factors, subsector_group)

            output_file = scaled_scenario_dir / year_file
            scaled_df.to_csv(output_file, compression='gzip', index=False)
            scenario_data[year] = scaled_df

        summary_data[scenario] = scenario_data
        generate_summary_file(scenario, scenario_data, scaled_scenario_dir)

    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scale 8760 load profiles based on scaling inputs')
    parser.add_argument('--input-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unscaled_shapes', 'shape_outputs'),
                        help=(
                            'Directory containing per-scenario, per-year unscaled shape CSVs. '
                            'Layout: <input-dir>/<scenario>/<year>.csv.gz. '
                            'Produce these by running step2 once per year, then use directly '
                            'if you want summary_shapes.csv output. '
                            'For the full v2 pipeline, use step3_scale_shapes.py instead.'
                        ))
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaled_shapes'),
                        help='Directory to store scaled shape outputs')
    parser.add_argument('--scaling-inputs', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaling_inputs_MWh.csv'),
                        help='CSV file with scaling inputs')
    args = parser.parse_args()
    main(args)
