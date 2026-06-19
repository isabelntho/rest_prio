# CLAUDE.md — rest_prio

Spatial multi-objective restoration prioritisation for the Canton of Bern, Switzerland.
Optimisation runs in **Python** (NSGA-III via pymoo); visualisation and analysis in **R**.

---

## Environment

- **Python**: managed with pixi (`pixi.toml` / `pixi.lock`). Run scripts with `pixi run python <script>`.
- **R**: managed with renv (`renv.lock`). Restore with `renv::restore()` in R.
- **CRS**: EPSG:2056 (Swiss LV95) throughout.

---

## Key directories

| Path | Contents |
|---|---|
| `inputs/` | Input rasters (anomaly, cost, eligible pixels) |
| `Ben_robustness/` | Robustness analysis TIFs and output figures |
| `Core_optimisation/` | Python optimisation code and Ben Black's R figures script |
| `Documentation/` | Plot functions (`_plot_functions.R`), notebooks, grid-run outputs |
| `results_files/` | Pareto front outputs from optimisation runs |
| `figs/` | Saved figure outputs |

Network data root: `Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/`

---

## Key files

### R
- `Documentation/_plot_functions.R` — all plotting functions for the RFOP sensitivity workflow (~3000 lines).


### Python
- `Core_optimisation/` — main optimisation pipeline.
- `export_to_r.py` — exports optimisation outputs to R-readable formats.
- `run_dynamic.py` — runs the dynamic optimisation variant.

---

## Canton of Bern shapefile

```
Y:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp
```
Filter on `NAME == "Bern"` to isolate the canton.

## Please remember

- If writing R code, do not use "→", as it is not read properly by radian
