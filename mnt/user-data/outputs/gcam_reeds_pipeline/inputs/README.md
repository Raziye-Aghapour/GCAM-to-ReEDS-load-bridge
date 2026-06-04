# inputs/

This folder holds all input data files. It is **git-ignored** — files here
are never committed to the repository. Download them from Zenodo instead.

## Required files

| File | Description |
|------|-------------|
| `example_gcam_query.xlsx` | GCAM-USA electricity demand output (3 sheets: Buildling, Transportation, Industry) |
| `efs_base_unscaled.csv.gz` | EFS base hourly load shapes — all 14 subsectors × 8760 hours × 51 states |
| `load_factors.csv` | BA share file — maps each state to its Balancing Authorities with fractional shares |

## Optional files (year-specific replacement shapes)

If you have higher-fidelity shapes from dsgrid or EVI-Pro, place them here
using this exact naming convention:

| File pattern | Replaces |
|---|---|
| `resstock_unscaled_texas_{year}.csv.gz` | Residential subsectors |
| `comstock_unscaled_texas_{year}.csv.gz` | Commercial subsectors |
| `evipro_ldv_unscaled_texas_{year}.csv.gz` | Transportation LDV subsector |

Where `{year}` is a GCAM snapshot year, e.g. `2030`, `2035`, etc.

## Zenodo DOI

<!-- Add your Zenodo DOI here once published -->
DOI: `10.5281/zenodo.XXXXXXX`
