import numpy as np
import pandas as pd
from pathlib import Path
import openpyxl

RNG = np.random.default_rng(42)
OUT = Path("synthetic_inputs")
OUT.mkdir(exist_ok=True)

WEATHER_YEAR = 2012
states3 = ["texas", "california", "new york"]

UCS_STATE_COLUMNS = [
    "alabama","alaska","arizona","arkansas","california","colorado",
    "connecticut","delaware","district of columbia","florida","georgia",
    "hawaii","idaho","illinois","indiana","iowa","kansas","kentucky",
    "louisiana","maine","maryland","massachusetts","michigan",
    "minnesota","mississippi","missouri","montana","nebraska","nevada",
    "new hampshire","new jersey","new mexico","new york",
    "north carolina","north dakota","ohio","oklahoma","oregon",
    "pennsylvania","rhode island","south carolina","south dakota",
    "tennessee","texas","utah","vermont","virginia","washington",
    "west virginia","wisconsin","wyoming",
]

# 8760-hour axis: 2012 leap year, Feb 29 retained, Dec 31 dropped
all_hours = pd.date_range("2012-01-01", periods=8784, freq="h")
drop_dec31 = (all_hours.month == 12) & (all_hours.day == 31)
hours_8760 = all_hours[~drop_dec31]
assert len(hours_8760) == 8760

SUBSECTORS = [
    ("residential",    "residential space heating and cooling"),
    ("residential",    "residential water heating"),
    ("residential",    "residential clothes and dish washing/drying"),
    ("residential",    "residential other"),
    ("commercial",     "commercial space heating and cooling"),
    ("commercial",     "commercial water heating"),
    ("commercial",     "commercial other"),
    ("industrial",     "industrial machine drives"),
    ("industrial",     "industrial other"),
    ("industrial",     "industrial process heat"),
    ("transportation", "transportation light-duty vehicles"),
    ("transportation", "transportation medium-duty trucks"),
    ("transportation", "transportation heavy-duty trucks"),
    ("transportation", "transportation other"),
]

hour_of_day = np.array(hours_8760.hour)
month_arr   = np.array(hours_8760.month)

all_rows = []
for sector, subsector in SUBSECTORS:
    base     = np.ones(8760)
    diurnal  = 0.3 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/3)
    seasonal = 0.2 * np.sin(2 * np.pi * (month_arr - 1) / 12)
    noise    = 0.05 * RNG.standard_normal(8760)
    shape_v  = np.abs(base + diurnal + seasonal + noise)
    shape_v /= shape_v.sum()

    rec = {
        "sector": sector,
        "subsector": subsector,
        "weather_datetime": hours_8760.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for st in UCS_STATE_COLUMNS:
        if st in states3:
            jitter = 1 + 0.1 * RNG.standard_normal(8760)
            rec[st] = shape_v * np.abs(jitter) * 1e6
        else:
            rec[st] = 0.0
    all_rows.append(pd.DataFrame(rec))

efs = pd.concat(all_rows, ignore_index=True)
efs_path = OUT / "efs_base_unscaled.csv.gz"
efs.to_csv(efs_path, index=False, compression="gzip")
print(f"Wrote {efs_path}  shape={efs.shape}")
per_sub = efs.groupby("subsector").size().unique().tolist()
print(f"  Subsectors={efs['subsector'].nunique()}, rows/subsector={per_sub}")

# ── GCAM xlsx ─────────────────────────────────────────────────────────────────
EJ_PER_MWH = 1.0 / 2.7777778e8
GCAM_ABBR  = {"texas": "TX", "california": "CA", "new york": "NY"}
YEAR       = 2030

gcam_def = {
    ("Buildling","buildings","resid heating"):         {"texas":30, "california":20, "new york":15},
    ("Buildling","buildings","resid cooling"):         {"texas":60, "california":25, "new york":10},
    ("Buildling","buildings","resid hot water"):       {"texas":20, "california":15, "new york":10},
    ("Buildling","buildings","resid clothes dryers"):  {"texas":8,  "california":6,  "new york":4},
    ("Buildling","buildings","resid clothes washers"): {"texas":3,  "california":2,  "new york":2},
    ("Buildling","buildings","resid dishwashers"):     {"texas":2,  "california":2,  "new york":1},
    ("Buildling","buildings","resid lighting"):        {"texas":12, "california":9,  "new york":6},
    ("Buildling","buildings","resid computers"):       {"texas":4,  "california":5,  "new york":3},
    ("Buildling","buildings","resid cooking"):         {"texas":3,  "california":3,  "new york":2},
    ("Buildling","buildings","resid freezers"):        {"texas":2,  "california":1,  "new york":1},
    ("Buildling","buildings","resid furnace fans"):    {"texas":1,  "california":1,  "new york":1},
    ("Buildling","buildings","resid refrigerators"):   {"texas":5,  "california":4,  "new york":3},
    ("Buildling","buildings","resid televisions"):     {"texas":4,  "california":3,  "new york":2},
    ("Buildling","buildings","resid other"):           {"texas":5,  "california":4,  "new york":3},
    ("Buildling","buildings","comm heating"):          {"texas":25, "california":20, "new york":12},
    ("Buildling","buildings","comm cooling"):          {"texas":55, "california":30, "new york":12},
    ("Buildling","buildings","comm hot water"):        {"texas":8,  "california":6,  "new york":4},
    ("Buildling","buildings","comm lighting"):         {"texas":30, "california":22, "new york":15},
    ("Buildling","buildings","comm cooking"):          {"texas":3,  "california":3,  "new york":2},
    ("Buildling","buildings","comm refrigeration"):    {"texas":8,  "california":6,  "new york":4},
    ("Buildling","buildings","comm ventilation"):      {"texas":10, "california":8,  "new york":5},
    ("Buildling","buildings","comm office"):           {"texas":5,  "california":5,  "new york":4},
    ("Buildling","buildings","comm non-building"):     {"texas":2,  "california":2,  "new york":1},
    ("Buildling","buildings","comm other"):            {"texas":4,  "california":3,  "new york":2},
    ("Transportation","transportation","trn_pass_road_LDV_4W"): {"texas":30,"california":40,"new york":15},
    ("Transportation","transportation","trn_freight_road"):      {"texas":25,"california":20,"new york":10},
    ("Transportation","transportation","trn_aviation_intl"):     {"texas":5, "california":8, "new york":5},
    ("Transportation","transportation","trn_shipping_intl"):     {"texas":3, "california":4, "new york":2},
    # rollup rows (step1 should drop these — double-counting guard)
    ("Transportation","transportation","trn_pass"):          {"texas":99,"california":99,"new york":99},
    ("Transportation","transportation","trn_pass_road"):     {"texas":99,"california":99,"new york":99},
    ("Transportation","transportation","trn_pass_road_LDV"): {"texas":99,"california":99,"new york":99},
    ("Transportation","transportation","trn_freight"):       {"texas":99,"california":99,"new york":99},
    ("Industry","industrial","other industrial energy use"):  {"texas":80,"california":50,"new york":25},
}

SHEET_GROUPS = {}
for (sheet, family, sub), state_vals in gcam_def.items():
    SHEET_GROUPS.setdefault(sheet, [])
    for state, twh in state_vals.items():
        mwh  = twh * 1e6
        ej_v = mwh * EJ_PER_MWH
        SHEET_GROUPS[sheet].append({
            "region": GCAM_ABBR[state], "sector": family,
            "subsector": sub, "input": "electricity",
            "scenario": "Reference", YEAR: ej_v,
        })

wb = openpyxl.Workbook()
first = True
for sheet_name, sheet_rows in SHEET_GROUPS.items():
    ws = wb.active if first else wb.create_sheet(sheet_name)
    if first: ws.title = sheet_name; first = False
    df_s = pd.DataFrame(sheet_rows)
    ws.append(list(df_s.columns))
    for _, r in df_s.iterrows():
        ws.append(list(r))

xlsx_path = OUT / "example_gcam_query.xlsx"
wb.save(xlsx_path)
print(f"Wrote {xlsx_path}")

# ── load_factors.csv ──────────────────────────────────────────────────────────
lf = pd.DataFrame([
    {"state":"texas",      "ba":"p_tx",  "share":0.10},
    {"state":"texas",      "ba":"erco",  "share":0.90},
    {"state":"california", "ba":"p_ca",  "share":1.00},
    {"state":"new york",   "ba":"p_ny",  "share":1.00},
])
lf_path = OUT / "load_factors.csv"
lf.to_csv(lf_path, index=False)
print(f"Wrote {lf_path}")
print("\nAll synthetic inputs ready in ./synthetic_inputs/")
