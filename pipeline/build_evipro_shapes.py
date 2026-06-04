"""
build_evipro_shapes.py
======================

Builds year-specific EVI-Pro LDV unscaled shape CSVs for Texas.
Output files feed into step2_combine_unscaled_shapes.py.

What it does:
  1. Downloads inputs from Zenodo (token required — record is restricted)
  2. Reads Texas daily temperatures from temperature_celsius-ba.h5
  3. Reads GCAM BEV fleet size from example_gcam_query.xlsx (NoEV sheet)
  4. Calls the NLR EVI-Pro Lite API for each day × temperature × year
  5. Blends C1 (unmanaged) and C2 (managed) charging shapes per YEAR_PARAMS
  6. Writes evipro_ldv_unscaled_texas_{year}.csv.gz for each GCAM year
  7. Writes a manifest JSON beside each output recording assumptions

Outputs (written to --output-dir):
  evipro_ldv_unscaled_texas_{year}.csv.gz
  evipro_ldv_unscaled_texas_{year}.manifest.json

Usage:
  python build_evipro_shapes.py \
      --evipro-api-key  YOUR_NLR_API_KEY \
      --zenodo-token    YOUR_ZENODO_TOKEN \
      --output-dir      /content/evipro_work/outputs \
      --workdir         /content/evipro_work \
      --years           2025,2030,2035,2040,2045,2050
"""
from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import time
import zipfile
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import requests

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('evipro')

# ── Constants ─────────────────────────────────────────────────────────────────

ZENODO_RECORD  = '19720340'
EFS_ZIP_NAME   = 'Medium_efs_unscaledshape.zip'
EFS_BIN_YEARS  = [2018, 2020, 2024, 2030, 2040, 2050]

SUPPORTED_GCAM_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
WEATHER_YEAR         = 2012   # EFS convention: keep Feb 29, drop Dec 31 → 8760 hours
ERCOT_BAS            = ['p48','p57','p59','p60','p61','p62','p63','p64','p65','p66','p67']
TARGET_STATE         = 'texas'

# [VERIFIED] from live Swagger spec + live API tests May 2026
EVIPRO_ENDPOINT            = 'https://developer.nlr.gov/api/evi-pro-lite/v1/daily-load-profile'
EVIPRO_ALLOWED_TEMPS       = np.array([40, 30, 20, 10, 0, -10, -20], dtype=int)
EVIPRO_CHARGER_COLS        = ['home_l1', 'home_l2', 'work_l1', 'work_l2', 'public_l2', 'public_l3']
EVIPRO_VALID_RES_CHARGING  = {'min_delay', 'max_delay', 'load_leveling', 'timed_charging'}
EVIPRO_VALID_WORK_CHARGING = {'min_delay', 'max_delay', 'load_leveling'}
EVIPRO_MIN_FLEET_SIZE      = 10_000
EVIPRO_MAX_FLEET_SIZE      = 10_000_000

# [VERIFIED] mean_dvmt=45 matches Acharya et al. 2024 Table I
# [ASSUMPTION] 45 mi/day chosen as highest available; TX VMT > national average
EVIPRO_MEAN_DVMT = 45

# [VERIFIED] Acharya et al. 2024 Table I
# [ASSUMPTION] Double-discontinuity at 2035: both pref_dist (Home80→Home60)
# AND managed_share (0%→30%) change simultaneously.
YEAR_PARAMS: Dict[int, Tuple[str, float, str]] = {
    2025: ('Home80', 0.00, '[ASSUMPTION] near-term: negligible managed charging'),
    2030: ('Home80', 0.00, '[ASSUMPTION]'),
    2035: ('Home60', 0.30, '[VERIFIED] Acharya et al. 2024 Table I'),
    2040: ('Home60', 0.50, '[INFERENCE] interpolated'),
    2045: ('Home60', 0.60, '[INFERENCE] interpolated'),
    2050: ('Home60', 0.70, '[VERIFIED] Acharya et al. 2024 Table I'),
}

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


# ── Year validation ───────────────────────────────────────────────────────────

def validate_years(years: List[int]) -> None:
    """Raise ValueError for any year not in YEAR_PARAMS."""
    bad = [y for y in years if y not in YEAR_PARAMS]
    if bad:
        raise ValueError(
            f'Unsupported GCAM years: {bad}.\n'
            f'Supported years (must have entries in YEAR_PARAMS): '
            f'{sorted(YEAR_PARAMS.keys())}'
        )


# ── Zenodo download ───────────────────────────────────────────────────────────

def download_zenodo(
    filename: str,
    out_path: Path,
    zenodo_token: str,
    max_retries: int = 3,
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
    for attempt in range(1, max_retries + 1):
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=300, stream=True)
                if r.status_code == 200:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                    log.info('[downloaded] %s  (%.1f MB)',
                             out_path.name, out_path.stat().st_size / 1e6)
                    return out_path
                log.warning('HTTP %s from %s', r.status_code, url)
            except Exception as e:
                log.warning('attempt %d — %s: %s', attempt, url, e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f'Failed to download {filename} from Zenodo {ZENODO_RECORD} '
                       f'after {max_retries} retries')


def unzip_efs_shapes(zip_path: Path, out_dir: Path) -> Dict[int, Path]:
    """Unzip Medium_efs_unscaledshape.zip → {year: path}.
    [VERIFIED] Contents: 2018.csv.gz, 2020.csv.gz, 2024.csv.gz,
                          2030.csv.gz, 2040.csv.gz, 2050.csv.gz
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    already = {
        int(p.stem.replace('.csv', '')): p
        for p in out_dir.glob('*.csv.gz')
        if p.stem.replace('.csv', '').isdigit()
    }
    if len(already) >= len(EFS_BIN_YEARS):
        log.info('[cached] EFS shapes already unzipped: %s', sorted(already.keys()))
        return already

    log.info('Unzipping %s ...', zip_path.name)
    year_paths: Dict[int, Path] = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        log.info('  Contents: %s', zf.namelist())
        for member in zf.namelist():
            fname = Path(member).name
            try:
                yr = int(fname.replace('.csv.gz', '').replace('.csv', ''))
            except ValueError:
                continue
            out_p = out_dir / f'{yr}.csv.gz'
            with zf.open(member) as src, open(out_p, 'wb') as dst:
                dst.write(src.read())
            year_paths[yr] = out_p
            log.info('  extracted %s', fname)
    return year_paths


# ── Temperature loading ───────────────────────────────────────────────────────

def load_texas_daily_temp(
    h5_path: Path,
    ercot_bas: List[str],
    year: int = 2012,
) -> pd.DataFrame:
    """Read ReEDS hourly BA temperatures → Texas daily mean, 365 rows.

    [VERIFIED] 2012 is a leap year (366 days in H5).
    [FIX leap-year] EFS axis keeps Feb 29 and drops Dec 31, NOT the other way around.
    daily_temps must match: keep Feb 29, drop Dec 31 (the last day of the year).
    This ensures build_annual() produces a profile whose 8760 positions align
    exactly with HOURLY_AXIS (Jan 1 00:00 → Dec 30 23:00, Feb 29 included).
    Without this fix: positions 1368–8759 (84% of the year) carry data from the
    next calendar day — a systematic 24-hour shift in the EV load profile.
    [ASSUMPTION] Average of 11 ERCOT BAs represents Texas statewide temperature.
    """
    log.info('Reading temperature H5: %s', h5_path.name)
    with h5py.File(h5_path, 'r') as f:
        ba_names = [c.decode() if isinstance(c, bytes) else c for c in f['columns'][:]]
        missing  = [b for b in ercot_bas if b not in ba_names]
        if missing:
            raise RuntimeError(f'ERCOT BAs not found in H5: {missing}')
        yr_key, idx_key = str(year), f'index_{year}'
        if yr_key not in f:
            raise RuntimeError(
                f'Year {year} not found in H5 file {h5_path.name}. '
                f'Available keys: {list(f.keys())}'
            )
        raw_idx = f[idx_key][:]
        ts_idx  = pd.to_datetime(
            raw_idx.astype(str) if raw_idx.dtype.kind in ('S', 'O') else raw_idx
        )
        data = f[yr_key][:]

    df_hourly = pd.DataFrame(data, index=ts_idx, columns=ba_names)
    tx_hourly = df_hourly[ercot_bas].mean(axis=1)
    tx_daily  = tx_hourly.resample('D').mean().reset_index()
    tx_daily.columns = ['date', 'temperature']

    # [FIX leap-year] Drop Dec 31, keep Feb 29 — matches EFS HOURLY_AXIS convention
    dec31 = (
        (pd.to_datetime(tx_daily['date']).dt.month == 12) &
        (pd.to_datetime(tx_daily['date']).dt.day   == 31)
    )
    if dec31.any():
        tx_daily = tx_daily[~dec31].reset_index(drop=True)

    n_days = len(tx_daily)
    if n_days != 365:
        raise ValueError(
            f'Expected 365 days after removing Dec 31, got {n_days}. '
            f'Check that WEATHER_YEAR={year} is a leap year and that '
            f'the H5 file covers the full calendar year.'
        )

    # Alignment invariant: Feb 29 must be present, Dec 31 must be absent
    feb29_present = (
        (pd.to_datetime(tx_daily['date']).dt.month == 2) &
        (pd.to_datetime(tx_daily['date']).dt.day   == 29)
    ).any()
    dec31_absent = not (
        (pd.to_datetime(tx_daily['date']).dt.month == 12) &
        (pd.to_datetime(tx_daily['date']).dt.day   == 31)
    ).any()

    if not feb29_present:
        raise ValueError(
            'Feb 29 is missing from daily_temps. '
            'EFS HOURLY_AXIS keeps Feb 29 and drops Dec 31. '
            'Verify WEATHER_YEAR is a leap year and drop logic targets Dec 31.'
        )
    if not dec31_absent:
        raise ValueError(
            'Dec 31 is still present in daily_temps after filtering. '
            'EFS HOURLY_AXIS ends on Dec 30 23:00 — Dec 31 must be dropped.'
        )

    tx_daily['temp_c'] = tx_daily['temperature'].map(
        lambda t: int(EVIPRO_ALLOWED_TEMPS[np.abs(EVIPRO_ALLOWED_TEMPS - t).argmin()])
    ).astype(int)
    tx_daily['weekday'] = pd.to_datetime(tx_daily['date']).dt.weekday

    log.info('Texas %d temperatures: %d days', year, len(tx_daily))
    log.info('  Raw range: %.1f°C – %.1f°C',
             tx_daily['temperature'].min(), tx_daily['temperature'].max())
    bin_summary = ' | '.join(
        f'{t}°C:{n}d'
        for t, n in tx_daily['temp_c'].value_counts().sort_index().items()
    )
    log.info('  EVI-Pro bins: %s', bin_summary)
    return tx_daily


# ── GCAM fleet sizing ─────────────────────────────────────────────────────────

def read_gcam_bev_fleet(xlsx_path: Path, state: str = 'TX') -> pd.DataFrame:
    """Read BEV pass-km from NoEV sheet → estimated fleet size per GCAM year.

    [VERIFIED] Formula: Acharya et al. 2024 Section III-A
    [ASSUMPTION] pass-km / vehicle-km assumes occupancy = 1.0.
    Pipeline outputs are unaffected (shape normalized to sum=1.0).
    """
    log.info('Reading GCAM BEV fleet from %s (sheet=NoEV)', xlsx_path.name)
    df = pd.read_excel(xlsx_path, sheet_name='NoEV')
    df.columns = [str(c).strip() for c in df.columns]
    bev = df[(df['region'] == state) & (df['technology'] == 'BEV')].copy()
    if bev.empty:
        raise RuntimeError(
            f'No BEV rows found in NoEV sheet for region={state!r}. '
            f'Available regions: {sorted(df["region"].unique())}'
        )

    year_cols = [c for c in df.columns if c.isdigit() and len(c) == 4]
    avg_km_yr = EVIPRO_MEAN_DVMT * 1.609 * 365

    rows = []
    for yr_str in [str(y) for y in SUPPORTED_GCAM_YEARS if str(y) in year_cols]:
        mpkm      = float(bev[yr_str].values[0])
        fleet_est = int(mpkm * 1e6 / avg_km_yr)
        api_fleet = max(min(fleet_est, EVIPRO_MAX_FLEET_SIZE), EVIPRO_MIN_FLEET_SIZE)
        rows.append({
            'year'           : int(yr_str),
            'bev_pass_km_M'  : mpkm,
            'fleet_estimated': fleet_est,
            'api_fleet_size' : api_fleet,
        })

    result = pd.DataFrame(rows)
    log.info('BEV fleet estimates for %s (avg %.0f km/vehicle/year):', state, avg_km_yr)
    log.info('  [ASSUMPTION] occupancy=1.0 → fleet is ~20%% overestimate; '
             'shape is normalized so pipeline output is unaffected')
    for _, row in result.iterrows():
        fe   = row['fleet_estimated']
        af   = row['api_fleet_size']
        note = ' [capped at API max]'  if fe > EVIPRO_MAX_FLEET_SIZE  else \
               ' [clamped to API min]' if fe < EVIPRO_MIN_FLEET_SIZE  else ''
        log.info('  %d: bev_pass_km=%.0f M | fleet_est=%d | api_fleet=%d%s',
                 row['year'], row['bev_pass_km_M'], fe, af, note)
    return result


# ── EVI-Pro API ───────────────────────────────────────────────────────────────

def validate_scenario(params: Dict) -> Dict:
    out = dict(params)
    out['temp_c'] = int(EVIPRO_ALLOWED_TEMPS[
        np.abs(EVIPRO_ALLOWED_TEMPS - float(out.get('temp_c', 20))).argmin()])
    fs = int(out.get('fleet_size', EVIPRO_MIN_FLEET_SIZE))
    if not (EVIPRO_MIN_FLEET_SIZE <= fs <= EVIPRO_MAX_FLEET_SIZE):
        raise ValueError(
            f'fleet_size={fs} is outside the allowed range '
            f'[{EVIPRO_MIN_FLEET_SIZE}, {EVIPRO_MAX_FLEET_SIZE}]'
        )
    out['fleet_size'] = fs
    checks = {
        'mean_dvmt'        : {25, 35, 45},
        'pev_type'         : {'PHEV20', 'PHEV50', 'BEV100', 'BEV250'},
        'pev_dist'         : {'BEV', 'PHEV', 'EQUAL'},
        'class_dist'       : {'Sedan', 'SUV', 'Equal'},
        'home_access_dist' : {'HA100', 'HA75', 'HA50'},
        'home_power_dist'  : {'MostL1', 'MostL2', 'Equal'},
        'work_power_dist'  : {'MostL1', 'MostL2', 'Equal'},
        'pref_dist'        : {'Home60', 'Home80', 'Home100'},
        'res_charging'     : EVIPRO_VALID_RES_CHARGING,
        'work_charging'    : EVIPRO_VALID_WORK_CHARGING,
    }
    for key, allowed in checks.items():
        if out.get(key) not in allowed:
            raise ValueError(
                f'Invalid scenario parameter: {key}={out.get(key)!r}. '
                f'Allowed values: {sorted(allowed)}'
            )
    return out


def api_call(params: Dict, evipro_api_key: str) -> Dict:
    p = validate_scenario(params)
    r = requests.get(
        EVIPRO_ENDPOINT,
        params={'api_key': evipro_api_key, **p},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f'EVI-Pro API returned HTTP {r.status_code}. '
            f'Response: {r.text[:300]}'
        )
    js = r.json()
    if 'results' not in js:
        raise RuntimeError(
            f'EVI-Pro API response missing "results" key. '
            f'Full response: {js}'
        )
    return js


_CACHE: Dict = {}

def get_profiles(params: Dict, evipro_api_key: str) -> Dict[str, pd.DataFrame]:
    params    = validate_scenario(params)
    cache_key = tuple(sorted(params.items()))
    if cache_key not in _CACHE:
        js = api_call(params, evipro_api_key)
        def to_df(obj):
            df = pd.DataFrame({c: obj.get(c, [0.] * 96) for c in EVIPRO_CHARGER_COLS})
            for c in EVIPRO_CHARGER_COLS:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.)
            df['total_kw'] = df[EVIPRO_CHARGER_COLS].sum(axis=1)
            return df
        _CACHE[cache_key] = {
            'weekday': to_df(js['results']['weekday_load_profile']),
            'weekend': to_df(js['results']['weekend_load_profile']),
        }
        time.sleep(0.15)
    return _CACHE[cache_key]


def build_annual(
    scenario      : Dict,
    daily_temps   : pd.DataFrame,
    evipro_api_key: str,
    label         : str = '',
    gcam_year     : int = 0,
) -> pd.DataFrame:
    """Build annual 15-min profile from daily EVI-Pro API calls.
    All timestamps in 'dt' are from WEATHER_YEAR (2012).
    gcam_year is stored as a label column for auditing.
    """
    frames = []
    for _, row in daily_temps.iterrows():
        p        = {**scenario, 'temp_c': int(row['temp_c'])}
        day_type = 'weekday' if int(row['weekday']) < 5 else 'weekend'
        day_df   = get_profiles(p, evipro_api_key)[day_type].copy()
        day_df['dt']        = pd.date_range(pd.Timestamp(row['date']), periods=96, freq='15min')
        day_df['label']     = label
        day_df['gcam_year'] = gcam_year
        frames.append(day_df)
    out    = pd.concat(frames, ignore_index=True)
    n_rows = len(out)
    if n_rows != 365 * 96:
        raise ValueError(
            f'build_annual: expected {365*96} rows (365 days × 96 15-min intervals), '
            f'got {n_rows}. Check that daily_temps has exactly 365 rows.'
        )
    return out


def to_hourly(annual_15min: pd.DataFrame, hourly_axis: pd.DatetimeIndex) -> pd.Series:
    """15-min kW → hourly average kW, indexed by hourly_axis.

    [FIX #1/#9] Index swap: groupby aggregates using EVI-Pro 2012 timestamps,
    then index is replaced with hourly_axis (EFS axis).
    Correctness relies on both being sorted chronologically ascending.
    """
    if not isinstance(hourly_axis, pd.DatetimeIndex):
        raise TypeError(
            f'hourly_axis must be a pd.DatetimeIndex, got {type(hourly_axis)}'
        )
    if not hourly_axis.is_monotonic_increasing:
        raise ValueError('hourly_axis must be chronologically sorted (monotonic increasing)')

    df         = annual_15min.copy()
    df['hour'] = pd.to_datetime(df['dt']).dt.floor('h')
    hourly     = df.groupby('hour')['total_kw'].mean().sort_index()

    if len(hourly) != 8760:
        raise ValueError(
            f'to_hourly: expected 8760 hourly values, got {len(hourly)}. '
            f'Check that annual_15min covers exactly 365 days.'
        )
    hourly.index = hourly_axis
    hourly.name  = 'total_kw'
    return hourly


# ── EFS LDV base loading ──────────────────────────────────────────────────────

def read_efs_ldv_base(efs_path: Path) -> pd.DataFrame:
    """Load EFS LDV base shape — used for non-TX state preservation."""
    log.info('Loading EFS LDV base from %s', efs_path.name)
    compression = 'gzip' if str(efs_path).lower().endswith('.gz') else None
    usecols     = ['sector', 'subsector', 'weather_datetime'] + UCS_STATE_COLUMNS
    df          = pd.read_csv(efs_path, compression=compression, usecols=usecols)
    ldv         = df[df['subsector'] == 'transportation light-duty vehicles'].copy()
    if ldv.empty:
        raise RuntimeError(
            f'No rows found for subsector "transportation light-duty vehicles" '
            f'in {efs_path.name}. Available subsectors: {sorted(df["subsector"].unique())}'
        )
    ldv['_dt'] = pd.to_datetime(ldv['weather_datetime'])
    ldv        = ldv.sort_values('_dt').reset_index(drop=True)
    if len(ldv) != 8760:
        raise ValueError(
            f'EFS LDV base must have exactly 8760 rows (one per hour); '
            f'got {len(ldv)} in {efs_path.name}'
        )
    return ldv.drop(columns=['_dt'])


# ── Output writer ─────────────────────────────────────────────────────────────

def write_ucs_evipro(
    hourly_shape  : pd.Series,
    gcam_year     : int,
    outdir        : Path,
    efs_ldv_base  : pd.DataFrame,
    fleet_df      : pd.DataFrame,
    efs_base_year : int,
    target_state  : str = 'texas',
) -> Path:
    """Write normalized LDV shape in UCS unscaled format + manifest JSON.

    [FIX #7] Non-TX states preserved from EFS LDV base.
    Integrity check: non-TX state sums must match EFS source.
    """
    s_norm = hourly_shape.astype(float).to_numpy()
    s_norm = s_norm / s_norm.sum()

    dt_values = efs_ldv_base['weather_datetime'].values
    non_tx    = [st for st in UCS_STATE_COLUMNS if st != target_state]

    df = pd.DataFrame({
        'sector'          : 'transportation',
        'subsector'       : 'transportation light-duty vehicles',
        'weather_datetime': dt_values,
    })

    # Preserve non-TX states from EFS base
    for st in UCS_STATE_COLUMNS:
        df[st] = pd.to_numeric(efs_ldv_base[st], errors='coerce').fillna(0.0).values

    # [FIX #7] Integrity check: non-TX values must match EFS source
    efs_non_tx_sum = float(
        pd.to_numeric(efs_ldv_base[non_tx].stack(), errors='coerce').fillna(0.0).sum()
    )
    out_non_tx_sum = float(df[non_tx].sum().sum())
    if abs(out_non_tx_sum - efs_non_tx_sum) >= 1e-3:
        raise RuntimeError(
            f'Non-TX state values corrupted for year {gcam_year}: '
            f'expected sum={efs_non_tx_sum:.4f}, got {out_non_tx_sum:.4f}. '
            f'This is a bug — check that EFS base columns are copied correctly.'
        )

    # Overwrite TX with EVI-Pro shape
    df[target_state] = s_norm

    # Normalization check
    tx_sum = float(df[target_state].sum())
    if abs(tx_sum - 1.0) >= 1e-8:
        raise RuntimeError(
            f'TX column for year {gcam_year} is not normalized: sum={tx_sum:.12f}. '
            f'Expected sum=1.0 (float64 precision).'
        )
    if len(df) != 8760:
        raise RuntimeError(
            f'Output for year {gcam_year} has {len(df)} rows, expected 8760.'
        )

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'evipro_ldv_unscaled_{target_state}_{gcam_year}.csv.gz'
    df.to_csv(out_path, index=False, compression='gzip')

    fs        = fleet_df[fleet_df['year'] == gcam_year]['api_fleet_size'].values[0]
    n_nonzero = sum(1 for st in non_tx if df[st].sum() > 0)
    pref, managed_share, note = YEAR_PARAMS[gcam_year]

    log.info('Wrote %s  (%.1f MB)', out_path.name, out_path.stat().st_size / 1e6)
    log.info('  fleet=%.2fM | TX sum=%.8f | non-TX states with EFS values: %d/%d',
             fs / 1e6, tx_sum, n_nonzero, len(non_tx))

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        'script'              : 'build_evipro_shapes.py',
        'generated_utc'       : datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'output_file'         : out_path.name,
        'gcam_year'           : gcam_year,
        'target_state'        : target_state,
        'weather_year'        : WEATHER_YEAR,
        'efs_base_year'       : efs_base_year,
        'calendar_convention' : 'keep Feb 29, drop Dec 31 → 8760 hours',
        'zenodo_record'       : ZENODO_RECORD,
        'api_fleet_size'      : int(fs),
        'pref_dist'           : pref,
        'c2_managed_share'    : managed_share,
        'c1_strategy'         : 'min_delay',
        'c2_strategy'         : 'load_leveling',
        'work_charging'       : 'min_delay',
        'home_access_dist'    : 'HA75',
        'mean_dvmt_miles'     : EVIPRO_MEAN_DVMT,
        'non_tx_states_source': f'EFS base year {efs_base_year}',
        'normalization'       : 'TX column sums to 1.0 (unitless shape)',
        'tx_column_sum'       : round(tx_sum, 12),
        'rows'                : len(df),
        'assumptions'         : [
            'occupancy=1.0 → fleet_estimated is ~20% overestimate; '
            'shape is normalized so pipeline output is unaffected',
            note,
            '[ASSUMPTION] work_charging fixed at min_delay for all years '
            '(Acharya et al. 2024 Table I)',
            '[ASSUMPTION] HA75: 75% of BEVs have home charging access',
        ],
    }
    manifest_path = outdir / f'evipro_ldv_unscaled_{target_state}_{gcam_year}.manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info('  manifest → %s', manifest_path.name)

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--evipro-api-key',
                   help='NLR EVI-Pro Lite API key (or set EVIPRO_API_KEY env var)')
    p.add_argument('--zenodo-token',
                   help='Zenodo personal access token (or set ZENODO_TOKEN env var)')
    p.add_argument('--output-dir', default='/content/evipro_work/outputs',
                   help='Directory for output CSV files (default: /content/evipro_work/outputs)')
    p.add_argument('--workdir',    default='/content/evipro_work',
                   help='Working directory for downloaded inputs (default: /content/evipro_work)')
    p.add_argument('--years',      default='2025,2030,2035,2040,2045,2050',
                   help=f'Comma-separated GCAM years to build. '
                        f'Supported: {sorted(YEAR_PARAMS.keys())}')
    args = p.parse_args()

    # ── Parse and validate years ──────────────────────────────────────────────
    try:
        years_to_build = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    except ValueError as e:
        raise ValueError(f'--years must be comma-separated integers: {e}') from e
    validate_years(years_to_build)   # raises ValueError for unsupported years

    # ── Credentials ───────────────────────────────────────────────────────────
    evipro_api_key = (
        args.evipro_api_key
        or os.environ.get('EVIPRO_API_KEY', '').strip()
        or getpass('Paste NLR EVI-Pro API key: ').strip()
    )
    if not evipro_api_key:
        raise RuntimeError('EVI-Pro API key is required. Pass --evipro-api-key or set EVIPRO_API_KEY.')

    zenodo_token = (
        args.zenodo_token
        or os.environ.get('ZENODO_TOKEN', '').strip()
        or getpass('Paste Zenodo token (Enter to skip if public): ').strip()
    )

    workdir = Path(args.workdir)
    outdir  = Path(args.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info('=' * 60)
    log.info('build_evipro_shapes.py  started')
    log.info('  years        : %s', years_to_build)
    log.info('  workdir      : %s', workdir)
    log.info('  output-dir   : %s', outdir)
    log.info('  weather_year : %d  (keep Feb 29, drop Dec 31)', WEATHER_YEAR)
    log.info('=' * 60)

    # ── Download inputs ───────────────────────────────────────────────────────
    log.info('--- Downloading inputs from Zenodo record %s ---', ZENODO_RECORD)
    gcam_xlsx_path = download_zenodo(
        'example_gcam_query.xlsx', workdir / 'gcam.xlsx', zenodo_token)
    temp_h5_path   = download_zenodo(
        'temperature_celsius-ba.h5', workdir / 'temperature_celsius-ba.h5', zenodo_token)
    efs_zip_path   = download_zenodo(
        EFS_ZIP_NAME, workdir / EFS_ZIP_NAME, zenodo_token)

    efs_shape_dir  = workdir / 'efs_shapes'
    efs_year_paths = unzip_efs_shapes(efs_zip_path, efs_shape_dir)
    log.info('EFS years available: %s', sorted(efs_year_paths.keys()))

    # ── Load EFS LDV base ─────────────────────────────────────────────────────
    efs_base_year = min(efs_year_paths.keys())
    log.info('--- Loading EFS LDV base (year %d) ---', efs_base_year)
    efs_ldv_base  = read_efs_ldv_base(efs_year_paths[efs_base_year])
    hourly_axis   = pd.DatetimeIndex(pd.to_datetime(efs_ldv_base['weather_datetime']))

    if len(hourly_axis) != 8760:
        raise ValueError(
            f'HOURLY_AXIS has {len(hourly_axis)} entries, expected 8760. '
            f'Check EFS LDV base file for year {efs_base_year}.'
        )
    if not hourly_axis.is_monotonic_increasing:
        raise ValueError(
            'HOURLY_AXIS is not monotonically increasing. '
            'EFS LDV base timestamps must be in chronological order.'
        )
    log.info('HOURLY_AXIS: %s → %s  (%d hours)',
             hourly_axis[0], hourly_axis[-1], len(hourly_axis))

    # ── Load temperatures and fleet sizes ─────────────────────────────────────
    log.info('--- Loading Texas daily temperatures ---')
    daily_temps = load_texas_daily_temp(temp_h5_path, ERCOT_BAS, WEATHER_YEAR)

    log.info('--- Loading GCAM BEV fleet sizes ---')
    fleet_df = read_gcam_bev_fleet(gcam_xlsx_path, state='TX')

    # ── Base API scenario (fixed across all years) ────────────────────────────
    BASE = {
        'mean_dvmt'       : EVIPRO_MEAN_DVMT,
        'pev_type'        : 'BEV250',
        'pev_dist'        : 'BEV',
        'class_dist'      : 'Equal',
        'home_access_dist': 'HA75',
        'home_power_dist' : 'MostL2',
        'work_power_dist' : 'MostL2',
        # [ASSUMPTION] work_charging fixed at min_delay for all years.
        # Rationale: Acharya et al. 2024 Table I sets work_charging=min_delay.
        'work_charging'   : 'min_delay',
    }

    # ── Build shapes per year ─────────────────────────────────────────────────
    _CACHE.clear()
    output_paths: Dict[int, Path] = {}

    log.info('--- Building EVI-Pro shapes for years: %s ---', years_to_build)
    for gcam_year in years_to_build:
        pref, managed_share, note = YEAR_PARAMS[gcam_year]
        fleet_row  = fleet_df[fleet_df['year'] == gcam_year]
        if fleet_row.empty:
            raise RuntimeError(
                f'No fleet data found for GCAM year {gcam_year}. '
                f'Check that example_gcam_query.xlsx NoEV sheet contains '
                f'year {gcam_year} for region TX.'
            )
        fleet_size = int(fleet_row.iloc[0]['api_fleet_size'])

        log.info('')
        log.info('=== GCAM year %d | fleet=%d | pref=%s | C2=%.0f%% ===',
                 gcam_year, fleet_size, pref, managed_share * 100)
        log.info('  %s', note)

        s_base = {**BASE, 'fleet_size': fleet_size, 'pref_dist': pref}
        s_c1   = {**s_base, 'res_charging': 'min_delay'}
        s_c2   = {**s_base, 'res_charging': 'load_leveling'}

        log.info('  Building C1 (min_delay) ...')
        h_c1      = to_hourly(build_annual(s_c1, daily_temps, evipro_api_key, 'C1', gcam_year), hourly_axis)
        h_c1_norm = h_c1 / h_c1.sum()

        if managed_share > 0:
            log.info('  Building C2 (load_leveling) ...')
            h_c2      = to_hourly(build_annual(s_c2, daily_temps, evipro_api_key, 'C2', gcam_year), hourly_axis)
            h_c2_norm = h_c2 / h_c2.sum()
            # [FIXED v6] Blend normalized shapes, not raw kW
            shape_blend = (1 - managed_share) * h_c1_norm + managed_share * h_c2_norm
            log.info('  Blended: %.0f%% C1 + %.0f%% C2',
                     (1 - managed_share) * 100, managed_share * 100)
        else:
            shape_blend = h_c1_norm.copy()
            log.info('  No managed charging (C1 only)')

        shape_blend      = shape_blend / shape_blend.sum()
        shape_blend.name = 'total_kw'

        lf = shape_blend.mean() / shape_blend.max()
        log.info('  Shape: LF=%.3f | peak @ %s | sum=%.12f',
                 lf, shape_blend.idxmax(), shape_blend.sum())

        output_paths[gcam_year] = write_ucs_evipro(
            shape_blend, gcam_year, outdir,
            efs_ldv_base, fleet_df, efs_base_year, TARGET_STATE,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info('')
    log.info('=' * 60)
    log.info('EVI-Pro shapes complete.')
    for yr, path in sorted(output_paths.items()):
        log.info('  %d: %s  (%.1f MB)', yr, path.name, path.stat().st_size / 1e6)
    log.info('All output files written to: %s', outdir)
    log.info('=' * 60)


if __name__ == '__main__':
    main()
