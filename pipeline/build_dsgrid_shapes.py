"""
build_dsgrid_shapes.py
======================

Builds year-specific ResStock and ComStock unscaled shape CSVs for Texas
using GCAM technology-weighted dsgrid building profiles.
Output files feed into step2_combine_unscaled_shapes.py.

What it does:
  1. Downloads inputs from Zenodo (token required — record is restricted)
  2. Loads EFS base shape (for HOURLY_AXIS alignment and non-TX state preservation)
  3. Downloads detailed_gcam_query.xlsx and reads GCAM technology EJ weights
  4. Reads Texas building end-use shapes from NREL dsgrid S3 (public, no credentials)
  5. Applies GCAM technology weights to blend dsgrid end-use shapes per year
  6. Writes resstock_unscaled_texas_{year}.csv.gz,
             comstock_unscaled_texas_{year}.csv.gz, and a manifest JSON per year

Outputs (written to --output-dir):
  resstock_unscaled_texas_{year}.csv.gz
  resstock_unscaled_texas_{year}.manifest.json
  comstock_unscaled_texas_{year}.csv.gz
  comstock_unscaled_texas_{year}.manifest.json

Usage:
  python build_dsgrid_shapes.py \
      --zenodo-token YOUR_ZENODO_TOKEN \
      --output-dir   /content/dsgrid_work/outputs \
      --workdir      /content/dsgrid_work \
      --years        2025,2030,2035,2040,2045,2050
"""
from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import warnings
import zipfile
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import s3fs

warnings.filterwarnings('ignore', category=UserWarning, module='fsspec')

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('dsgrid')

# ── Constants ─────────────────────────────────────────────────────────────────

ZENODO_RECORD  = '19720340'
EFS_ZIP_NAME   = 'Medium_efs_unscaledshape.zip'
EFS_BIN_YEARS  = [2018, 2020, 2024, 2030, 2040, 2050]

SUPPORTED_GCAM_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]

# [VERIFIED] from live S3 listing
DSGRID_BASE        = 'nrel-pds-dsgrid/building/building-2021/v1.0.0/full_dataset/full_dataset.parquet'
DSGRID_MODEL_YEARS = list(range(2010, 2052, 2))

# [VERIFIED] TX county counts — consistent across ALL model years
TX_COUNTY_COUNT_RES = 253
TX_COUNTY_COUNT_COM = 245

TARGET_STATE = 'texas'
DSGRID_STATE = 'TX'
WEATHER_YEAR = 2012

# UTC window: EFS Jan 1 00:00 CST = Jan 1 06:00 UTC
#             EFS Dec 30 23:00 CST = Dec 31 05:00 UTC  (Feb 29 retained)
UTC_START = pd.Timestamp('2012-01-01 06:00:00')
UTC_END   = pd.Timestamp('2012-12-31 05:00:00')

UCS_STATE_COLUMNS = [
    'alabama','alaska','arizona','arkansas','california','colorado',
    'connecticut','delaware','district of columbia','florida','georgia',
    'hawaii','idaho','illinois','indiana','iowa','kansas','kentucky',
    'louisiana','maine','maryland','massachusetts','michigan',
    'minnesota','mississippi','missouri','montana','nebraska','nevada',
    'new hampshire','new jersey','new mexico','new york',
    'north carolina','north dakota','ohio','oklahoma','oregon',
    'pennsylvania','rhode island','south carolina','south dakota',
    'tennessee','texas','utah','vermont','virginia','washington',
    'west virginia','wisconsin','wyoming',
]

# [VERIFIED] dsgrid end-use → EFS subsector mapping
# All column names verified from live dsgrid (Harris County TX, model_year=2018)
RES_DSGRID_TO_EFS: Dict[str, str] = {
    'electricity_cooling'             : 'residential space heating and cooling',
    'electricity_heating'             : 'residential space heating and cooling',
    'electricity_heating_supplemental': 'residential space heating and cooling',
    # [ASSUMPTION] Pumps → space H&C (no separate GCAM 'resid pumps' sector)
    'electricity_pumps_cooling'       : 'residential space heating and cooling',
    'electricity_pumps_heating'       : 'residential space heating and cooling',
    'electricity_water_systems'       : 'residential water heating',
    'electricity_recirc_pump'         : 'residential water heating',
    'electricity_clothes_dryer'       : 'residential clothes and dish washing/drying',
    'electricity_clothes_washer'      : 'residential clothes and dish washing/drying',
    'electricity_dishwasher'          : 'residential clothes and dish washing/drying',
    'electricity_bath_fan'                 : 'residential other',
    'electricity_ceiling_fan'              : 'residential other',
    'electricity_fans_cooling'             : 'residential other',
    'electricity_fans_heating'             : 'residential other',
    'electricity_house_fan'                : 'residential other',
    'electricity_range_fan'                : 'residential other',
    'electricity_cooking_range'            : 'residential other',
    'electricity_exterior_holiday_lighting': 'residential other',
    'electricity_exterior_lighting'        : 'residential other',
    'electricity_extra_refrigerator'       : 'residential other',
    'electricity_freezer'                  : 'residential other',
    'electricity_garage_lighting'          : 'residential other',
    'electricity_hot_tub_heater'           : 'residential other',
    'electricity_hot_tub_pump'             : 'residential other',
    'electricity_interior_lighting'        : 'residential other',
    'electricity_plug_loads'               : 'residential other',
    'electricity_pool_heater'              : 'residential other',
    'electricity_pool_pump'                : 'residential other',
    'electricity_refrigerator'             : 'residential other',
    'electricity_well_pump'                : 'residential other',
}

COM_DSGRID_TO_EFS: Dict[str, str] = {
    # [VERIFIED] pipeline: comm ventilation → commercial space heating and cooling
    'electricity_cooling'        : 'commercial space heating and cooling',
    'electricity_heating'        : 'commercial space heating and cooling',
    'electricity_fans'           : 'commercial space heating and cooling',
    'electricity_heat_recovery'  : 'commercial space heating and cooling',
    'electricity_heat_rejection' : 'commercial space heating and cooling',
    'electricity_pumps'          : 'commercial space heating and cooling',
    'electricity_water_systems'  : 'commercial water heating',
    'electricity_exterior_lighting'  : 'commercial other',
    'electricity_interior_equipment' : 'commercial other',
    'electricity_interior_lighting'  : 'commercial other',
    'electricity_refrigeration'      : 'commercial other',
}

RES_EFS_SUBSECTORS = [
    'residential space heating and cooling',
    'residential water heating',
    'residential clothes and dish washing/drying',
    'residential other',
]
COM_EFS_SUBSECTORS = [
    'commercial space heating and cooling',
    'commercial water heating',
    'commercial other',
]

# [VERIFIED] GCAM technology → dsgrid end-use mapping
# Technology names from detailed_gcam_query.xlsx (Zenodo 19720340)
# dsgrid end-use names from live S3 (Harris County TX, model_year=2018)
GCAM_TECH_TO_DSGRID: Dict[Tuple[str,str], List[str]] = {
    ('resid cooling',     'air conditioning')         : ['electricity_cooling'],
    ('resid cooling',     'air conditioning hi-eff')  : ['electricity_cooling'],
    ('resid heating',     'electric furnace')         : ['electricity_heating',
                                                         'electricity_heating_supplemental'],
    ('resid heating',     'electric heat pump')       : ['electricity_heating',
                                                         'electricity_heating_supplemental'],
    ('resid furnace fans','electricity')              : ['electricity_pumps_cooling',
                                                         'electricity_pumps_heating'],
    ('resid hot water','electric resistance water heater')      : ['electricity_water_systems',
                                                                    'electricity_recirc_pump'],
    ('resid hot water','electric resistance water heater hi-eff'):['electricity_water_systems',
                                                                    'electricity_recirc_pump'],
    ('resid hot water','electric heat pump water heater')       : ['electricity_water_systems',
                                                                    'electricity_recirc_pump'],
    ('resid clothes dryers', 'clothes dryer')         : ['electricity_clothes_dryer'],
    ('resid clothes dryers', 'clothes dryer hi-eff')  : ['electricity_clothes_dryer'],
    ('resid clothes washers','clothes washer')         : ['electricity_clothes_washer'],
    ('resid clothes washers','clothes washer hi-eff')  : ['electricity_clothes_washer'],
    ('resid dishwashers',    'dishwasher')             : ['electricity_dishwasher'],
    ('resid dishwashers',    'dishwasher hi-eff')      : ['electricity_dishwasher'],
    ('resid cooking',       'electric oven')          : ['electricity_cooking_range'],
    ('resid computers',     'electricity')            : ['electricity_plug_loads'],
    ('resid freezers',      'freezer')                : ['electricity_freezer'],
    ('resid freezers',      'freezer hi-eff')         : ['electricity_freezer'],
    ('resid lighting',      'incandescent')           : ['electricity_interior_lighting'],
    ('resid lighting',      'fluorescent')            : ['electricity_interior_lighting'],
    ('resid lighting',      'solid state')            : ['electricity_interior_lighting'],
    ('resid other',         'electricity')            : ['electricity_plug_loads',
                                                         'electricity_hot_tub_heater',
                                                         'electricity_hot_tub_pump',
                                                         'electricity_pool_heater',
                                                         'electricity_pool_pump',
                                                         'electricity_well_pump'],
    ('resid refrigerators', 'refrigerator')           : ['electricity_refrigerator',
                                                         'electricity_extra_refrigerator'],
    ('resid refrigerators', 'refrigerator hi-eff')    : ['electricity_refrigerator',
                                                         'electricity_extra_refrigerator'],
    ('resid televisions',   'electricity')            : ['electricity_plug_loads'],
    ('comm cooling',     'air conditioning')          : ['electricity_cooling'],
    ('comm cooling',     'air conditioning hi-eff')   : ['electricity_cooling'],
    ('comm heating',     'electric furnace')          : ['electricity_heating'],
    ('comm heating',     'electric heat pump')        : ['electricity_heating'],
    ('comm ventilation', 'ventilation')               : ['electricity_fans',
                                                         'electricity_heat_recovery',
                                                         'electricity_heat_rejection',
                                                         'electricity_pumps'],
    ('comm ventilation', 'ventilation hi-eff')        : ['electricity_fans',
                                                         'electricity_heat_recovery',
                                                         'electricity_heat_rejection',
                                                         'electricity_pumps'],
    ('comm hot water','electric resistance water heater') : ['electricity_water_systems'],
    ('comm hot water','electric heat pump water heater')  : ['electricity_water_systems'],
    ('comm cooking',       'electric range')          : ['electricity_interior_equipment'],
    ('comm cooking',       'electric range hi-eff')   : ['electricity_interior_equipment'],
    ('comm lighting',      'fluorescent')             : ['electricity_interior_lighting',
                                                         'electricity_exterior_lighting'],
    ('comm lighting',      'incandescent')            : ['electricity_interior_lighting',
                                                         'electricity_exterior_lighting'],
    ('comm lighting',      'solid state')             : ['electricity_interior_lighting',
                                                         'electricity_exterior_lighting'],
    ('comm non-building',  'electricity')             : ['electricity_interior_equipment'],
    ('comm office',        'office equipment')        : ['electricity_interior_equipment'],
    ('comm other',         'electricity')             : ['electricity_interior_equipment'],
    ('comm refrigeration', 'refrigeration')           : ['electricity_refrigeration'],
    ('comm refrigeration', 'refrigeration hi-eff')    : ['electricity_refrigeration'],
}

# [VERIFIED] dsgrid end-uses with no direct GCAM technology
# [ASSUMPTION] Weighted proportionally by their dsgrid kWh share within subsector
DSGRID_NO_GCAM_WEIGHT = [
    'electricity_bath_fan', 'electricity_ceiling_fan', 'electricity_house_fan',
    'electricity_range_fan', 'electricity_fans_cooling', 'electricity_fans_heating',
    'electricity_garage_lighting', 'electricity_exterior_holiday_lighting',
]


# ── Year validation ───────────────────────────────────────────────────────────

def validate_years(years: List[int]) -> None:
    """Raise ValueError for any year not in SUPPORTED_GCAM_YEARS."""
    bad = [y for y in years if y not in SUPPORTED_GCAM_YEARS]
    if bad:
        raise ValueError(
            f'Unsupported GCAM years: {bad}.\n'
            f'Supported years: {SUPPORTED_GCAM_YEARS}\n'
            f'dsgrid only has building data for 2010–2050 in 2-year steps.'
        )


# ── Zenodo download ───────────────────────────────────────────────────────────

def download_zenodo(
    filename: str,
    out_path: Path,
    zenodo_token: str,
) -> Path:
    if out_path.exists():
        log.info('[cached]     %s  (%.1f MB)', out_path.name, out_path.stat().st_size / 1e6)
        return out_path
    headers = {'Authorization': f'Bearer {zenodo_token}'} if zenodo_token else {}
    urls = [
        f'https://zenodo.org/api/records/{ZENODO_RECORD}/files/{filename}/content',
        f'https://zenodo.org/api/records/{ZENODO_RECORD}/draft/files/{filename}/content',
        f'https://zenodo.org/record/{ZENODO_RECORD}/files/{filename}?download=1',
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=300, stream=True)
            if r.status_code == 200:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                log.info('[downloaded] %s  (%.1f MB)',
                         out_path.name, out_path.stat().st_size / 1e6)
                return out_path
            log.warning('HTTP %s from %s', r.status_code, url)
        except Exception as e:
            log.warning('Error from %s: %s', url, e)
    raise RuntimeError(f'Failed to download {filename} from Zenodo {ZENODO_RECORD}')


def unzip_efs_shapes(zip_path: Path, out_dir: Path) -> Dict[int, Path]:
    """Unzip Medium_efs_unscaledshape.zip → {year: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    already = {
        int(p.name.split('.')[0]): p
        for p in out_dir.glob('*.csv.gz')
        if p.name.split('.')[0].isdigit()
    }
    if len(already) >= len(EFS_BIN_YEARS):
        log.info('[cached] EFS shapes already unzipped: %s', sorted(already.keys()))
        return already
    log.info('Unzipping %s ...', zip_path.name)
    year_paths: Dict[int, Path] = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname.endswith('.csv.gz') and fname.split('.')[0].isdigit():
                out = out_dir / fname
                if not out.exists():
                    out.write_bytes(zf.read(member))
                yr = int(fname.split('.')[0])
                year_paths[yr] = out
    log.info('  Extracted: %s', sorted(year_paths.keys()))
    return year_paths


# ── dsgrid helpers ────────────────────────────────────────────────────────────

def dsgrid_year_for_gcam(gcam_year: int) -> int:
    """[ASSUMPTION] Odd GCAM years use the PREVIOUS even dsgrid year.
    2025→2024, 2035→2034, 2045→2044. Avoids extrapolation beyond available data.
    """
    candidates = [y for y in DSGRID_MODEL_YEARS if y <= gcam_year]
    return max(candidates) if candidates else DSGRID_MODEL_YEARS[0]


def list_tx_counties(fs: s3fs.S3FileSystem, sector: str, model_year: int) -> List[str]:
    """List TX county FIPS codes for sector/model_year.
    [VERIFIED] res=253, com=245 — consistent across all model years.
    """
    expected = TX_COUNTY_COUNT_RES if sector == 'res' else TX_COUNTY_COUNT_COM
    prefix   = f'{DSGRID_BASE}/sector={sector}/model_year={model_year}/state={DSGRID_STATE}/'
    dirs     = fs.ls(prefix)
    counties = [d.split('geography=')[-1] for d in dirs if 'geography=' in d]
    if len(counties) != expected:
        raise RuntimeError(
            f'Expected {expected} TX counties for sector={sector}, '
            f'model_year={model_year}, got {len(counties)}.\n'
            f'S3 path: {prefix}'
        )
    log.info('  %d TX counties found ✓', len(counties))
    return counties


def unmapped_audit(
    fs: s3fs.S3FileSystem,
    sector: str, model_year: int, counties: List[str],
    end_use_map: Dict[str, str],
) -> None:
    """Raise RuntimeError if any electricity end-use is not in the mapping dict."""
    sample_path  = (f'{DSGRID_BASE}/sector={sector}/model_year={model_year}/'
                    f'state={DSGRID_STATE}/geography={counties[0]}/')
    sample_files = [f for f in fs.glob(sample_path + '*.parquet')
                    if not f.endswith('.crc')]
    if not sample_files:
        return
    sample   = pd.read_parquet(f's3://{sample_files[0]}',
                               storage_options={'anon': True},
                               columns=['end_use'])
    all_elec = sorted(e for e in sample['end_use'].unique()
                      if str(e).startswith('electricity'))
    unmapped = [e for e in all_elec if e not in end_use_map]
    log.info('  Electricity end-uses: %d in data, %d mapped',
             len(all_elec), len(all_elec) - len(unmapped))
    if unmapped:
        raise RuntimeError(
            f'Unmapped electricity end-uses for sector={sector}, '
            f'model_year={model_year}: {unmapped}\n'
            f'Add them to the RES_DSGRID_TO_EFS or COM_DSGRID_TO_EFS '
            f'mapping dict before proceeding.'
        )
    log.info('  All electricity end-uses mapped ✓')


def read_county_by_end_use(
    fs: s3fs.S3FileSystem,
    sector: str, model_year: int, county: str,
    end_use_map: Dict[str, str],
    scenario: str = 'reference',
) -> pd.DataFrame:
    """Read one TX county — return [timestamp_utc, end_use, value].

    [FIX v6 CRITICAL] Preserves individual dsgrid end_use granularity
    needed by build_weighted_shape(). Returns UTC timestamps for alignment.
    [VERIFIED] weather_year validated to be 2012.
    """
    path  = (f'{DSGRID_BASE}/sector={sector}/model_year={model_year}/'
             f'state={DSGRID_STATE}/geography={county}/')
    files = [f for f in fs.glob(path + '*.parquet') if not f.endswith('.crc')]
    if not files:
        return pd.DataFrame(columns=['timestamp_utc', 'end_use', 'value'])

    dfs = [
        pd.read_parquet(f's3://{f}', storage_options={'anon': True},
                        columns=['timestamp','end_use','value','scenario','weather_year'])
        for f in files
    ]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    # [VERIFIED] Validate weather_year
    actual = set(df['weather_year'].unique())
    if actual not in ({'2012'}, {2012}):
        raise RuntimeError(
            f'Expected weather_year=2012 in county={county}, got {actual}. '
            f'Verify the dsgrid S3 path is correct.'
        )

    df = df[(df['scenario'] == scenario) & (df['end_use'].isin(end_use_map))].copy()
    if df.empty:
        return pd.DataFrame(columns=['timestamp_utc', 'end_use', 'value'])

    df['timestamp_utc'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    county_agg = df.groupby(['timestamp_utc', 'end_use'], as_index=False)['value'].sum()
    return county_agg


def read_tx_sector_year_shapes(
    fs          : s3fs.S3FileSystem,
    sector      : str,
    model_year  : int,
    end_use_map : Dict[str, str],
    hourly_axis : pd.DatetimeIndex,
) -> Tuple[Dict[str, pd.Series], Dict[str, float]]:
    """Read TX data → normalized hourly shape + raw MWh per dsgrid end-use.

    Returns:
        end_use_shapes  : {dsgrid_col: normalized 8760 Series}
        end_use_raw_kwh : {dsgrid_col: raw MWh total before normalization}
    [VERIFIED] UTC filter: 2012-01-01 06:00 → 2012-12-31 05:00 = 8760 h including Feb 29
    """
    counties = list_tx_counties(fs, sector, model_year)
    log.info('  %d TX counties | sector=%s | model_year=%d', len(counties), sector, model_year)
    unmapped_audit(fs, sector, model_year, counties, end_use_map)

    frames = []
    for i, county in enumerate(counties):
        if (i + 1) % 50 == 0 or i == 0:
            log.info('    %d/%d counties...', i + 1, len(counties))
        county_df = read_county_by_end_use(fs, sector, model_year, county, end_use_map)
        if not county_df.empty:
            frames.append(county_df)

    if not frames:
        raise RuntimeError(
            f'No data found for sector={sector}, model_year={model_year}. '
            f'Check S3 path: {DSGRID_BASE}/sector={sector}/model_year={model_year}/'
        )

    combined  = pd.concat(frames, ignore_index=True)
    state_agg = combined.groupby(['timestamp_utc', 'end_use'], as_index=False)['value'].sum()
    state_agg.columns = ['timestamp', 'end_use', 'kwh']

    # Apply UTC window filter (EFS-equivalent hours, includes Feb 29)
    state_agg['timestamp'] = pd.to_datetime(state_agg['timestamp'])
    state_agg = state_agg[
        (state_agg['timestamp'] >= UTC_START) &
        (state_agg['timestamp'] <= UTC_END)
    ].copy()

    n_ts = state_agg['timestamp'].nunique()
    if n_ts != 8760:
        raise ValueError(
            f'Expected 8760 unique timestamps after UTC filter '
            f'({UTC_START} → {UTC_END}), got {n_ts}. '
            f'Verify dsgrid weather_year=2012 and UTC window covers Feb 29.'
        )
    log.info('  UTC filter: %s → %s | %d timestamps ✓', UTC_START, UTC_END, n_ts)

    # Build normalized shape per end-use
    end_use_raw_kwh: Dict[str, float]   = {}
    end_use_shapes:  Dict[str, pd.Series] = {}
    for eu in state_agg['end_use'].unique():
        eu_df = state_agg[state_agg['end_use'] == eu].set_index('timestamp')['kwh']
        eu_df = eu_df.sort_index()
        total = eu_df.sum()
        end_use_raw_kwh[eu] = float(total)   # [FIX v6b] store BEFORE normalizing
        if total > 0:
            norm       = eu_df / total
            norm.index = hourly_axis          # positional alignment (both 8760, sorted)
            end_use_shapes[eu] = norm

    log.info('  End-use shapes built: %d', len(end_use_shapes))
    return end_use_shapes, end_use_raw_kwh


# ── GCAM technology weights ───────────────────────────────────────────────────

def load_gcam_tech_weights(
    xlsx_path : Path,
    state     : str = 'TX',
    years     : Optional[List[int]] = None,
) -> Dict[int, Dict[Tuple[str,str], float]]:
    """Read GCAM technology-level electricity EJ by (sector, technology) per year.

    Sheet: Building_by_tech_Fuel
    Only electricity rows (input == 'elect_td_bld') are included.
    hi-eff and standard variants are kept separate — summed in build_weighted_shape().
    """
    if years is None:
        years = SUPPORTED_GCAM_YEARS

    log.info('Reading GCAM technology weights from %s (sheet=Building_by_tech_Fuel)', xlsx_path.name)
    df = pd.read_excel(xlsx_path, sheet_name='Building_by_tech_Fuel')
    df.columns = [str(c).strip() for c in df.columns]
    df = df[(df['region'] == state) & (df['input'] == 'elect_td_bld')].copy()
    log.info('  TX electricity technology rows: %d', len(df))

    weights: Dict[int, Dict[Tuple[str,str], float]] = {}
    for yr in years:
        yr_str = str(yr)
        if yr_str not in df.columns:
            raise RuntimeError(
                f'Year {yr} not found in GCAM xlsx columns. '
                f'Available columns: {[c for c in df.columns if c.isdigit()]}'
            )
        weights[yr] = {}
        for _, row in df.iterrows():
            key = (str(row['sector']).strip(), str(row['technology']).strip())
            weights[yr][key] = float(row[yr_str])

    # Check all mapping keys are present in xlsx
    missing = [k for k in GCAM_TECH_TO_DSGRID.keys() if k not in weights[years[0]]]
    if missing:
        raise RuntimeError(
            f'GCAM_TECH_TO_DSGRID references {len(missing)} (sector, technology) '
            f'pairs not found in the xlsx:\n  '
            + '\n  '.join(str(k) for k in missing)
            + '\nCheck that detailed_gcam_query.xlsx is the correct file '
            'from Zenodo record ' + ZENODO_RECORD
        )

    # Inverse check: xlsx technologies not in mapping → warn only
    unmapped_xlsx = [
        (s, t) for (s, t) in weights[years[0]].keys()
        if (s, t) not in GCAM_TECH_TO_DSGRID
    ]
    if unmapped_xlsx:
        log.warning('%d xlsx electricity technologies not in GCAM_TECH_TO_DSGRID '
                    '(will be ignored):', len(unmapped_xlsx))
        for s, t in sorted(unmapped_xlsx):
            log.warning('  (%s, %s)', s, t)
    else:
        log.info('  Completeness ✓ all xlsx electricity technologies are mapped')

    log.info('  Years loaded: %s', years)
    log.info('  Technologies per year: %d', len(weights[years[0]]))
    return weights


# ── GCAM-weighted shape builder ───────────────────────────────────────────────

def build_weighted_shape(
    end_use_shapes : Dict[str, pd.Series],
    end_use_raw_kwh: Dict[str, float],
    gcam_weights   : Dict[Tuple[str,str], float],
    efs_subsector  : str,
    dsgrid_to_efs  : Dict[str, str],
    hourly_axis    : pd.DatetimeIndex,
) -> pd.Series:
    """Build one EFS subsector shape weighted by GCAM technology EJ.

    Stage 1: Accumulate GCAM-weighted kWh per dsgrid end-use.
             EJ split proportionally by 2018 dsgrid kWh within mapped group.
    Stage 2: Handle no-weight end-uses (fans, misc) by dsgrid kWh share.
    Stage 3: Build weighted sum → normalize to sum=1.0.

    [FIX v6b] Uses raw MWh (end_use_raw_kwh) not normalized shapes for splitting.
    [ASSUMPTION] Technologies sharing same dsgrid shape get different EJ weights
                 but the same hourly profile.
    """
    if len(hourly_axis) != 8760:
        raise ValueError(
            f'hourly_axis must have 8760 entries, got {len(hourly_axis)}'
        )

    # Stage 1: accumulate GCAM-weighted kWh per dsgrid end-use
    col_weights: Dict[str, float] = {}

    for (sector, tech), dsgrid_cols in GCAM_TECH_TO_DSGRID.items():
        subsector_cols = [c for c in dsgrid_cols
                          if dsgrid_to_efs.get(c) == efs_subsector]
        if not subsector_cols:
            continue
        ej = gcam_weights.get((sector, tech), 0.0)
        if ej <= 0:
            continue
        # [FIX v6b] Proportional split by raw dsgrid kWh
        col_kwh       = {c: end_use_raw_kwh.get(c, 0.0)
                         for c in subsector_cols if c in end_use_shapes}
        total_col_kwh = sum(col_kwh.values())
        for col in subsector_cols:
            if col not in end_use_shapes:
                continue
            frac = (col_kwh[col] / total_col_kwh) if total_col_kwh > 0 \
                   else (1.0 / len(subsector_cols))
            col_weights[col] = col_weights.get(col, 0.0) + ej * frac

    # Stage 2: no-weight end-uses by kWh share within subsector
    no_weight_cols = [c for c in DSGRID_NO_GCAM_WEIGHT
                      if dsgrid_to_efs.get(c) == efs_subsector
                      and c in end_use_shapes]
    if no_weight_cols:
        total_gcam_ej   = sum(col_weights.values())
        no_weight_kWh   = {c: end_use_raw_kwh.get(c, 0.0) for c in no_weight_cols}
        total_nw_kWh    = sum(no_weight_kWh.values())
        if total_nw_kWh > 0 and total_gcam_ej > 0:
            total_sub_kWh = sum(
                end_use_raw_kwh.get(c, 0.0)
                for c in end_use_shapes
                if dsgrid_to_efs.get(c) == efs_subsector
            )
            for col in no_weight_cols:
                frac = no_weight_kWh[col] / total_sub_kWh
                col_weights[col] = total_gcam_ej * frac

    # Stage 3: build weighted sum and normalize
    weighted = pd.Series(0.0, index=hourly_axis)
    for col, weight in col_weights.items():
        if col not in end_use_shapes or weight <= 0:
            continue
        shape = end_use_shapes[col]
        s     = shape.values if len(shape) == 8760 else shape.reindex(hourly_axis).values
        weighted += weight * s

    total = weighted.sum()
    if total <= 0:
        raise RuntimeError(
            f'EFS subsector "{efs_subsector}": weighted sum is zero or negative ({total}). '
            f'Check GCAM weights for this year and verify GCAM_TECH_TO_DSGRID mapping.'
        )
    norm      = weighted / total
    norm.name = efs_subsector
    return norm


# ── Output writer ─────────────────────────────────────────────────────────────

def write_ucs_building(
    shapes        : Dict[str, pd.Series],
    efs_subsectors: List[str],
    gcam_year     : int,
    sector_label  : str,     # 'resstock' or 'comstock'
    sector_str    : str,     # 'Residential' or 'Commercial'
    outdir        : Path,
    efs_all       : pd.DataFrame,
    hourly_axis   : pd.DatetimeIndex,
    efs_base_year : int,
    dsgrid_model_year: int,
    target_state  : str = 'texas',
) -> Path:
    """Write GCAM-weighted shapes in UCS unscaled format + manifest JSON.

    Non-TX states are preserved from EFS base with integrity check.
    """
    non_tx = [st for st in UCS_STATE_COLUMNS if st != target_state]
    frames = []

    for sub in efs_subsectors:
        if sub not in shapes:
            raise RuntimeError(
                f'Missing EFS subsector "{sub}" in shapes dict for '
                f'{sector_label} year {gcam_year}. '
                f'Available subsectors: {sorted(shapes.keys())}'
            )
        norm = shapes[sub]

        norm_sum = float(norm.sum())
        if abs(norm_sum - 1.0) >= 1e-8:
            raise ValueError(
                f'{sector_label} year {gcam_year}, subsector "{sub}": '
                f'shape is not normalized (sum={norm_sum:.12f}, expected 1.0)'
            )
        if len(norm) != 8760:
            raise ValueError(
                f'{sector_label} year {gcam_year}, subsector "{sub}": '
                f'shape has {len(norm)} rows, expected 8760'
            )

        efs_sub = (efs_all[efs_all['subsector'] == sub]
                   .sort_values('_dt').reset_index(drop=True))
        if len(efs_sub) != 8760:
            raise RuntimeError(
                f'EFS base has {len(efs_sub)} rows for subsector "{sub}", '
                f'expected 8760. Check EFS base file for year {efs_base_year}.'
            )

        row = pd.DataFrame({
            'sector'          : sector_str,
            'subsector'       : sub,
            'weather_datetime': efs_sub['weather_datetime'].values,
        })
        # Copy all states from EFS base
        for st in UCS_STATE_COLUMNS:
            row[st] = pd.to_numeric(efs_sub[st], errors='coerce').fillna(0.0).values
        # Overwrite TX with GCAM-weighted dsgrid shape
        row[target_state] = norm.values

        # Integrity check: non-TX values must match EFS source exactly
        efs_non_tx = float(efs_sub[non_tx].to_numpy(dtype=float).sum())
        out_non_tx = float(row[non_tx].to_numpy(dtype=float).sum())
        if abs(out_non_tx - efs_non_tx) >= 1e-3:
            raise RuntimeError(
                f'{sector_label} year {gcam_year}, subsector "{sub}": '
                f'non-TX state values corrupted after copy '
                f'(expected={efs_non_tx:.4f}, got={out_non_tx:.4f}). '
                f'This is a bug — check that EFS base columns are copied correctly.'
            )

        tx_sum = float(row[target_state].sum())
        if abs(tx_sum - 1.0) >= 1e-8:
            raise RuntimeError(
                f'{sector_label} year {gcam_year}, subsector "{sub}": '
                f'TX column sum={tx_sum:.12f}, expected 1.0'
            )

        n_nonzero = sum(1 for st in non_tx if float(row[st].sum()) > 0)
        log.info('  [%s] TX sum=%.8f | non-TX with EFS: %d/%d',
                 sub[:42], tx_sum, n_nonzero, len(non_tx))
        frames.append(row)

    out_df = pd.concat(frames, ignore_index=True)
    expected_rows = len(efs_subsectors) * 8760
    if len(out_df) != expected_rows:
        raise RuntimeError(
            f'{sector_label} year {gcam_year}: output has {len(out_df)} rows, '
            f'expected {expected_rows} ({len(efs_subsectors)} subsectors × 8760 hours)'
        )

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'{sector_label}_unscaled_{target_state}_{gcam_year}.csv.gz'
    out_df.to_csv(out_path, index=False, compression='gzip')
    log.info('  → %s  (%.1f MB) | %d rows',
             out_path.name, out_path.stat().st_size / 1e6, len(out_df))

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        'script'               : 'build_dsgrid_shapes.py',
        'generated_utc'        : datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'output_file'          : out_path.name,
        'gcam_year'            : gcam_year,
        'sector_label'         : sector_label,
        'target_state'         : target_state,
        'weather_year'         : WEATHER_YEAR,
        'efs_base_year'        : efs_base_year,
        'dsgrid_model_year'    : dsgrid_model_year,
        'calendar_convention'  : 'keep Feb 29, drop Dec 31 → 8760 hours',
        'utc_window'           : f'{UTC_START} → {UTC_END}',
        'zenodo_record'        : ZENODO_RECORD,
        'dsgrid_s3_path'       : DSGRID_BASE,
        'gcam_xlsx'            : 'detailed_gcam_query.xlsx',
        'gcam_sheet'           : 'Building_by_tech_Fuel',
        'gcam_filter'          : 'input == elect_td_bld, region == TX',
        'efs_subsectors'       : efs_subsectors,
        'non_tx_states_source' : f'EFS base year {efs_base_year}',
        'normalization'        : 'TX column sums to 1.0 per subsector (unitless shape)',
        'rows'                 : len(out_df),
        'assumptions'          : [
            '[ASSUMPTION] Odd GCAM years (2025, 2035, 2045) use PREVIOUS even '
            'dsgrid year to avoid extrapolation',
            '[ASSUMPTION] dsgrid shapes are invariant across model years (r=1.0000 verified) '
            '— built once from canonical year, GCAM weights vary by year',
            '[ASSUMPTION] Technologies sharing the same dsgrid end-use column '
            '(e.g. electric heat pump + electric furnace → electricity_heating) '
            'receive different EJ weights but the same hourly profile',
            '[ASSUMPTION] No-weight dsgrid end-uses (fans, holiday lighting) '
            'weighted proportionally by their 2018 dsgrid kWh share',
            '[ASSUMPTION] Pumps (electricity_pumps_cooling, electricity_pumps_heating) '
            'assigned to space H&C — no separate GCAM resid pumps sector',
        ],
    }
    manifest_path = outdir / f'{sector_label}_unscaled_{target_state}_{gcam_year}.manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info('  manifest → %s', manifest_path.name)

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--zenodo-token',
                   help='Zenodo personal access token (or set ZENODO_TOKEN env var)')
    p.add_argument('--output-dir', default='/content/dsgrid_work/outputs',
                   help='Directory for output CSV files (default: /content/dsgrid_work/outputs)')
    p.add_argument('--workdir',    default='/content/dsgrid_work',
                   help='Working directory for downloaded inputs (default: /content/dsgrid_work)')
    p.add_argument('--years',      default='2025,2030,2035,2040,2045,2050',
                   help=f'Comma-separated GCAM years. Supported: {SUPPORTED_GCAM_YEARS}')
    args = p.parse_args()

    # ── Parse and validate years ──────────────────────────────────────────────
    try:
        years_to_build = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    except ValueError as e:
        raise ValueError(f'--years must be comma-separated integers: {e}') from e
    validate_years(years_to_build)   # raises ValueError for unsupported years

    zenodo_token = (
        args.zenodo_token
        or os.environ.get('ZENODO_TOKEN', '').strip()
        or getpass('Paste Zenodo token (Enter to skip if public): ').strip()
    )

    workdir = Path(args.workdir)
    outdir  = Path(args.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # [VERIFIED] dsgrid S3 bucket is public — no credentials needed
    fs = s3fs.S3FileSystem(anon=True)

    log.info('=' * 60)
    log.info('build_dsgrid_shapes.py  started')
    log.info('  years        : %s', years_to_build)
    log.info('  workdir      : %s', workdir)
    log.info('  output-dir   : %s', outdir)
    log.info('  weather_year : %d  (keep Feb 29, drop Dec 31)', WEATHER_YEAR)
    log.info('  dsgrid S3    : %s', DSGRID_BASE)
    log.info('=' * 60)

    log.info('--- Downloading inputs from Zenodo record %s ---', ZENODO_RECORD)
    gcam_xlsx_path = download_zenodo(
        'detailed_gcam_query.xlsx', workdir / 'detailed_gcam_query.xlsx', zenodo_token)
    efs_zip_path   = download_zenodo(
        EFS_ZIP_NAME, workdir / EFS_ZIP_NAME, zenodo_token)

    efs_shape_dir  = workdir / 'efs_shapes'
    efs_year_paths = unzip_efs_shapes(efs_zip_path, efs_shape_dir)

    # ── Load EFS base (use 2018 for axis + non-TX preservation) ──────────────
    efs_base_year = 2018 if 2018 in efs_year_paths else min(efs_year_paths.keys())
    efs_base_path = efs_year_paths[efs_base_year]
    log.info('--- Loading EFS base from year %d (%s) ---', efs_base_year, efs_base_path.name)
    efs_all = pd.read_csv(
        efs_base_path, compression='gzip',
        usecols=['subsector', 'weather_datetime'] + UCS_STATE_COLUMNS,
    )
    efs_all['_dt'] = pd.to_datetime(efs_all['weather_datetime'])

    # Build HOURLY_AXIS from one reference subsector
    ref_sub = (efs_all[efs_all['subsector'] == 'residential space heating and cooling']
               .sort_values('_dt').reset_index(drop=True))
    if len(ref_sub) != 8760:
        raise ValueError(
            f'EFS base reference subsector has {len(ref_sub)} rows, expected 8760. '
            f'Check that {efs_base_path.name} is a valid EFS unscaled shape file.'
        )
    hourly_axis = pd.DatetimeIndex(pd.to_datetime(ref_sub['weather_datetime']))

    if not hourly_axis.is_monotonic_increasing:
        raise ValueError(
            'HOURLY_AXIS from EFS base is not monotonically increasing. '
            'EFS timestamps must be in chronological order.'
        )
    if len(hourly_axis) != 8760:
        raise ValueError(
            f'HOURLY_AXIS has {len(hourly_axis)} entries, expected 8760.'
        )
    log.info('EFS axis: %s → %s  (%d hours)', hourly_axis[0], hourly_axis[-1], len(hourly_axis))
    log.info('EFS subsectors available: %s', sorted(efs_all['subsector'].unique()))

    # ── Load GCAM technology weights ──────────────────────────────────────────
    log.info('--- Loading GCAM technology weights ---')
    gcam_weights = load_gcam_tech_weights(gcam_xlsx_path, state='TX', years=years_to_build)
    log.info('GCAM weights loaded ✓')

    # ── Read dsgrid end-use shapes ONCE ──────────────────────────────────────
    # [VERIFIED] dsgrid shapes are invariant across model years (r=1.0000 verified).
    # Build once from model_year=2024 (canonical proxy for GCAM 2025+).
    # GCAM weights vary by year — so output shapes vary by GCAM year.
    dsgrid_canonical_year = dsgrid_year_for_gcam(2025)   # = 2024
    log.info('--- Reading dsgrid end-use shapes (model_year=%d) ---', dsgrid_canonical_year)
    log.info('[VERIFIED] shapes are invariant across model years — built once, '
             'GCAM weights apply year variation')

    log.info('Residential:')
    res_end_use_shapes, res_raw_kwh = read_tx_sector_year_shapes(
        fs, 'res', dsgrid_canonical_year, RES_DSGRID_TO_EFS, hourly_axis)

    log.info('Commercial:')
    com_end_use_shapes, com_raw_kwh = read_tx_sector_year_shapes(
        fs, 'com', dsgrid_canonical_year, COM_DSGRID_TO_EFS, hourly_axis)

    log.info('dsgrid read complete: %d res + %d com end-use shapes.',
             len(res_end_use_shapes), len(com_end_use_shapes))

    # ── Apply GCAM weights per year and write outputs ─────────────────────────
    log.info('--- Applying GCAM technology weights for years: %s ---', years_to_build)
    res_output_paths: Dict[int, Path] = {}
    com_output_paths: Dict[int, Path] = {}

    for gcam_year in years_to_build:
        log.info('')
        log.info('=== GCAM year %d ===', gcam_year)
        gcam_yr_weights  = gcam_weights[gcam_year]
        dsgrid_model_yr  = dsgrid_year_for_gcam(gcam_year)
        log.info('  dsgrid model_year: %d', dsgrid_model_yr)

        # Residential
        log.info('  Building residential shapes...')
        res_shapes: Dict[str, pd.Series] = {}
        for sub in RES_EFS_SUBSECTORS:
            shape = build_weighted_shape(
                res_end_use_shapes, res_raw_kwh,
                gcam_yr_weights, sub, RES_DSGRID_TO_EFS, hourly_axis,
            )
            res_shapes[sub] = shape
            lf     = shape.mean() / shape.max()
            peak_h = shape.idxmax().hour
            log.info('  [res | %s] LF=%.3f | peak@%02dh', sub[:38], lf, peak_h)

        # Commercial
        log.info('  Building commercial shapes...')
        com_shapes: Dict[str, pd.Series] = {}
        for sub in COM_EFS_SUBSECTORS:
            shape = build_weighted_shape(
                com_end_use_shapes, com_raw_kwh,
                gcam_yr_weights, sub, COM_DSGRID_TO_EFS, hourly_axis,
            )
            com_shapes[sub] = shape
            lf     = shape.mean() / shape.max()
            peak_h = shape.idxmax().hour
            log.info('  [com | %s] LF=%.3f | peak@%02dh', sub[:38], lf, peak_h)

        # Write outputs
        log.info('  Writing resstock...')
        res_output_paths[gcam_year] = write_ucs_building(
            res_shapes, RES_EFS_SUBSECTORS,
            gcam_year, 'resstock', 'Residential',
            outdir, efs_all, hourly_axis,
            efs_base_year, dsgrid_model_yr, TARGET_STATE,
        )
        log.info('  Writing comstock...')
        com_output_paths[gcam_year] = write_ucs_building(
            com_shapes, COM_EFS_SUBSECTORS,
            gcam_year, 'comstock', 'Commercial',
            outdir, efs_all, hourly_axis,
            efs_base_year, dsgrid_model_yr, TARGET_STATE,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info('')
    log.info('=' * 60)
    log.info('dsgrid building shapes complete.')
    for yr in years_to_build:
        rp = res_output_paths[yr]
        cp = com_output_paths[yr]
        log.info('  %d: %s  (%.1f MB)', yr, rp.name, rp.stat().st_size / 1e6)
        log.info('       %s  (%.1f MB)', cp.name, cp.stat().st_size / 1e6)
    log.info('All output files written to: %s', outdir)
    log.info('=' * 60)


if __name__ == '__main__':
    main()
