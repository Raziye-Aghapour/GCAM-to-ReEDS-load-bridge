# GCAM → ReEDS Load Shape Pipeline

Converts GCAM-USA state-level electricity demand projections into hourly load
shape files (`.h5`) that drop directly into the ReEDS capacity expansion model.

## What it does

```
GCAM xlsx  +  EFS / ResStock / ComStock / EVI-Pro shapes
        │
        ▼
  step 1 — GCAM xlsx → scaling_inputs_MWh.csv          (annual MWh targets per state)
  step 2 — shape sources → combined_unscaled_{year}.csv.gz   (hourly profiles, per year)
  step 3 — combined + targets → scaled_shapes/<scenario>/<year>.csv.gz
  step 4 — scaled shapes → <scenario>_load_hourly.h5    (ReEDS-ready)
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Create the conda environment (Apple Silicon / macOS)
conda env create -f environment.yml
conda activate gcam_reeds

# 3. Place input data in inputs/  (see inputs/README.md for file list and Zenodo DOI)
```

## Run the full pipeline

```bash
python pipeline/run_all.py \
    --gcam-xlsx     inputs/example_gcam_query.xlsx \
    --efs-base      inputs/efs_base_unscaled.csv.gz \
    --resstock-dir  inputs/ \
    --comstock-dir  inputs/ \
    --evipro-dir    inputs/ \
    --load-factors  inputs/load_factors.csv \
    --scenario      gcam_default \
    --gcam-scenario Reference \
    --years         2025,2030,2035,2040,2045,2050 \
    --output-h5     outputs/gcam_default_load_hourly.h5 \
    --workdir       pipeline_work
```

## Run the synthetic dry-run test (no real data needed)

Verifies the pipeline mechanics work end-to-end in under 2 minutes:

```bash
# Generate synthetic inputs
python tests/generate_synthetic_inputs.py

# Run all 4 steps on synthetic data
python pipeline/step1_gcam_to_scaling_inputs.py \
    --gcam-xlsx     synthetic_inputs/example_gcam_query.xlsx \
    --scenario      synthetic_test \
    --gcam-scenario Reference \
    --output        synthetic_inputs/scaling_inputs_MWh.csv \
    --years         2030

python pipeline/step2_combine_unscaled_shapes.py \
    --efs-base  synthetic_inputs/efs_base_unscaled.csv.gz \
    --output    synthetic_inputs/combined_unscaled_2030.csv.gz

python pipeline/step3_scale_shapes.py \
    --combined-unscaled-dir  synthetic_inputs \
    --scaling-inputs         synthetic_inputs/scaling_inputs_MWh.csv \
    --output-dir             synthetic_inputs/scaled_shapes \
    --scenario               synthetic_test \
    --years                  2030

python pipeline/step4_generate_h5.py \
    --scaled-dir   synthetic_inputs/scaled_shapes \
    --scenario     synthetic_test \
    --load-factors synthetic_inputs/load_factors.csv \
    --output       synthetic_inputs/synthetic_test_load_hourly.h5 \
    --years        2030
```

Expected output from step 3: `rel error=0.00e+00` (energy conserved to float precision).

## Pipeline scripts

| Script | Role |
|--------|------|
| `pipeline/step1_gcam_to_scaling_inputs.py` | Reads GCAM xlsx, outputs annual MWh targets per state |
| `pipeline/step2_combine_unscaled_shapes.py` | Combines EFS base with ResStock / ComStock / EVI-Pro |
| `pipeline/step3_scale_shapes.py` | Scales hourly profiles to match GCAM annual totals |
| `pipeline/step4_generate_h5.py` | Assembles ReEDS-format H5 file with BA disaggregation |
| `pipeline/run_all.py` | Orchestrator — runs steps 1–4 in sequence |
| `pipeline/main.py` | Standalone alternative to step 3 (also writes summary_shapes.csv) |
| `tests/generate_synthetic_inputs.py` | Builds minimal synthetic dataset for dry-run testing |

## Folder structure

```
gcam_reeds_pipeline/
├── pipeline/          ← the 6 pipeline scripts (in git)
├── tests/             ← synthetic data generator (in git)
├── inputs/            ← real data from Zenodo (git-ignored)
├── pipeline_work/     ← intermediate files, auto-created (git-ignored)
├── outputs/           ← final H5 files, auto-created (git-ignored)
├── snapshots/         ← rollback snapshots (git-ignored)
├── environment.yml
├── .gitignore
└── README.md
```

## Data

Input data is archived on Zenodo.  
DOI: `10.5281/zenodo.XXXXXXX`  
See `inputs/README.md` for the full file list.
