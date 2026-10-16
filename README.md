# Parameter-free spatial interpolation for sparse climate monitoring

Testing how far **parameter-free statistical interpolation** — natural-neighbour
(Voronoi/Sibson) interpolation and an altitude-aware 3-D extension — can go
against tuned inverse-distance weighting (IDW) *and* against machine-learning
baselines (kriging/GP, random forest, gradient boosting, MLP), for filling gaps
in sparse environmental sensor networks over complex terrain.

The study spans **three climate variables** on **real, sparse station networks**:
temperature and precipitation (the two pillars of climate) and PM2.5 air quality.
The organising question is not "which interpolator wins" but **"does it pay to
factor in elevation, and how do you know?"** — answered differently by each
variable, for reasons rooted in physics.

## Motivation

Climate adaptation needs dense, current environmental fields, yet ground sensors
are sparse — sparsest in the remote, high-relief terrain where they matter most.
Interpolation fills those gaps, but the accurate ML interpolators are costly to
run *continuously* at scale: kriging/GP is O(n³) to fit, and tree/DL models must
be refit (and periodically retuned) as observations stream in. That motivates
**parameter-free statistical interpolation**: no training, no tuning,
deterministic, and — as the timing benchmark shows — cost independent of network
size. The question is how much accuracy, if any, that cheapness costs.

## Methods

- **Projection** — lat/lon → metres via azimuthal-equidistant about the data
  centroid (`aqv/data.py`), so Euclidean distance approximates ground distance.
  No geospatial dependencies.
- **IDW** (baseline) — KD-tree neighbours, weight `1/dᵖ`; `p∈{1,2,3}`, `k∈{5,8,12}`
  tuned **per fold** by nested LOOCV over the training stations only (no evaluation
  leakage), so every comparison is against a per-fold-optimal IDW.
- **NNI** — **discrete Sibson** (`aqv/interpolators.py`): rasterise a grid around
  the query, measure how much Voronoi territory it steals from each site; the
  stolen fraction is the Sibson weight. No explicit Voronoi diagram, no geometric
  degeneracies, parameter-free, and it **generalises unchanged to 3-D** (voxels).
  Points outside the convex hull fall back to tuned IDW and are flagged.
- **NNI-3D (altitude tiles)** — anisotropic embedding `(x, y, β·z)`; interpolate
  in 3-D so elevation is a spatial coordinate.
- **NNI-residual** — regress the value on elevation (degree-1), interpolate the
  residual with 2-D NNI, add the trend back. The linear "deconstruction" of the
  3-D tile.
- **ML baselines** (`aqv/baselines.py`) — GaussianProcess (= kriging, RBF+White),
  RandomForest, HistGradientBoosting, MLP (target standardised via
  `TransformedTargetRegressor`), all given identical `(x, y, elev)`.
- **Protocol notes** — IDW power and neighbour count (and the 3-D β) are tuned
  *per fold* by nested LOOCV over training stations only, so the held-out station
  never informs the baseline; the MLP target is standardised so its L2 penalty is
  scale-free; the EPA loader falls back to the "Events Included" series for
  monitors that reported no event-free annual mean, which keeps smoke-affected
  monitors in wildfire years.
- **Validation** — leave-one-station-out CV; RMSE / MAE / bias / R² / MAPE;
  paired Wilcoxon signed-rank on per-station absolute errors; error stratified by
  distance-to-nearest-station (`aqv/validate.py`).

## Data (all real, sparse station networks)

| variable | source | elevation | loader |
|---|---|---|---|
| **Temperature** | GHCN-Monthly TAVG, annual mean | in station metadata | `aqv/ghcn.py` |
| **Precipitation** | GHCN-Daily PRCP → annual total | in station metadata | `aqv/precip.py` |
| **PM2.5** | EPA AQS annual mean (needs free key) | Open-Meteo (Copernicus GLO-90) join | `aqv/epa_aqs.py`, `aqv/elevation.py` |

Mountain regions with real relief and genuinely sparse coverage: Colorado,
Western US, Sierra Nevada, Pacific NW / Cascades, Alps, Himalaya, Caucasus, Andes
(temperature); Pacific NW, Sierra Nevada, Colorado (precipitation); Southern/
Northern Rockies, Sierra–Great Basin, Cascades (PM2.5), for 2023 and the 2021
wildfire year.

An earlier version used a satellite-derived PM2.5 product (ACAG) as ground truth.
That was dropped: it is a smooth CNN model output that already ingests topography,
so testing "does elevation help" on it was circular. All PM2.5 results here are
real EPA monitors. (ACAG returns as a *comparison baseline* in the satellite
head-to-head below.) The EPA loader prefers each monitor's "No Events" series and
falls back to "Events Included", so smoke-flagged monitors are retained in
wildfire years rather than silently dropped.

## Headline result — the three-tier elevation law

Best elevation-aware method vs tuned IDW (LOOCV RMSE reduction), by variable:

| variable | elev–value \|r\| | elevation gain | mechanism |
|---|---|---|---|
| **Temperature** | 0.80–0.98 | **+26% to +82%** (all p<0.01) | universal physical law (lapse rate) |
| **Precipitation** | 0.13–0.34 | **+5% to +12%** | orographic — real but *directional* |
| **PM2.5 / PM10** | 0.03–0.79 | **~0%** (elevation increment sig. in 1/16 — chance level) | no vertical law (source-driven) |

A variable with a universal vertical law gets huge gains from factoring in height;
one with directional elevation structure gets moderate gains; one with no vertical
structure gets nothing. The gains are **largest exactly where monitoring is
sparsest and relief is highest** — Alps +82%, Himalaya +76% — the remote-valley
scenario that motivates the work.

### The diagnostic that *doesn't* work (and the one that does)

The obvious a-priori test — "regress value on elevation, use it if |r| is large" —
**fails**, and both failure directions appear in the data:

- **Precipitation: |r| = 0.13 but elevation helps +12%.** The orographic effect is
  real but directional (windward ↑, rain-shadow ↓), so it cancels in a scalar
  correlation.
- **PM2.5 (N. Rockies): |r| = 0.74 but elevation helps ~0%.** The correlation is
  incidental (where a few clean high sites happen to sit), not exploitable
  structure.

So linear correlation is not the tell. What works: (1) the variable's **physics**
predicts the *ordering* (universal law > directional > none); and (2) because
parameter-free NNI is instant and needs no tuning, you **decide empirically —
run elevation-aware vs 2-D in cross-validation and keep the winner**, per variable
per region. The method's cheapness is precisely what lets you settle the question
by trying both, with no expensive model selection.

## Statistics vs ML/DL — competitive, and far cheaper to keep live

Same `(x, y, elevation)` for every learner. Colorado temperature (n=348), LOOCV:

| method | family | RMSE | R² | vs IDW |
|---|---|---|---|---|
| GP/kriging+elev | kernel | 1.321 | 0.896 | +30.7% |
| GradientBoosting | tree-ML | 1.333 | 0.894 | +30.0% |
| RandomForest | tree-ML | 1.341 | 0.892 | +29.6% |
| **NNI-residual (ours)** | geostat | **1.405** | 0.882 | +26.3% |
| **NNI-3D tiles (ours)** | geostat | **1.533** | 0.859 | +19.6% |
| MLP (target-standardised) | DL | 1.566 | 0.853 | +17.8% |

**Head-to-head (paired Wilcoxon vs NNI-residual):** GP/GBM/RF are nominally ~5–6%
ahead but **not significant** (p = 0.54–0.70). The y-standardised MLP is no longer
a strawman: it beats tuned IDW on all 8 temperature networks (Colorado 1.566), yet
is the single worst of 9 methods on 16 of 27 runs (uniformly on precip and PM10);
NNI-residual beats it on 24 of 27. Across all **27** region-years, training-free
geostatistics is statistically indistinguishable from the best tuned ML in **24**
— the 3 exceptions (temp/WesternUS 0.032, pm25/Cascades-2021 0.024,
pm10/SierraGreatBasin-2023 0.028) are chance-consistent under 27 comparisons.

### Timing — the real-time-at-scale argument

Seconds to ingest *n* stations and predict 300 targets (`run_timing.py`, Western
US temperature, median of 3):

| method | family | n=100 | n=300 | n=1000 | n=1943 |
|---|---|---|---|---|---|
| IDW | stats | 0.001 | 0.001 | 0.001 | 0.001 |
| NNI-2D | stats | 0.30 | 0.30 | 0.30 | 0.30 |
| NNI-3D | stats | 1.58 | 1.58 | 1.55 | 1.55 |
| GP/kriging | ML | 0.024 | 0.085 | 0.90 | **5.51** |
| RandomForest | ML | 0.20 | 0.20 | 0.21 | 0.26 |
| MLP | ML | 1.02 | 1.33 | 2.43 | 3.92 |

The honest reading: not "stats is always faster" (RandomForest is quick). Rather,
statistical interpolators need **no training or tuning** and their cost is
**independent of network size**, whereas the *most accurate* ML method
(GP/kriging) is **O(n³)** and becomes the bottleneck at continental scale, and the
tree/DL models must refit on every data update — for accuracy that is
statistically indistinguishable from parameter-free NNI.

## The air-quality counter-inference — "pollution reaches up"

The adage is "the higher you go, the cleaner the air." Real stations say otherwise.

- **Across the *inhabited* elevation range, PM2.5 falls only modestly** (pooled 301
  stations, 2023, fixed loader: 7.5 vs 6.7 µg/m³ between the 0–1000 m and
  1000–1500 m bands, where 82% of monitors sit — a ~10% drop while elevation
  triples). The weak overall r = −0.38 is steepened mainly by 14 very-high,
  mostly-unpopulated sites above 2000 m. (The earlier "flat, 6.2 vs 6.4" figure
  came from the pre-fix loader that dropped high-reading smoke-flagged monitors.)
- **In the 2021 wildfire year the relationship flattens or reverses.** Clean-year
  r = −0.27…−0.50 becomes −0.14, −0.11, and in the **Cascades flips to +0.51 —
  air is *dirtier* up high.** Smoke concentrates at all elevations, and
  elevation-detrending (which assumes cleaner-up-high) then *hurts* by 14%.

Unlike temperature, air quality has no reliable vertical law, so **you cannot
infer a mountain community's air quality from its altitude** — and in the smoke
events that most threaten those communities, being high offers no protection and
can be worse. Elevation is a valid shortcut for interpolating temperature; for air
quality it is not.

## Satellite head-to-head (ACAG V6.GL CNN)

Held-out truth = real EPA monitors (fixed loader), both predictors scored against it.
**Normal year (2023):** the decision-rule-selected training-free variant beats the
satellite in 3 of 4 regions, and the only statistically significant difference
anywhere favours interpolation (Sierra–Great Basin, p=0.004) — despite ACAG being
calibrated *on* these monitors. Tuned ML beats the satellite there too: it is the
data regime, not our estimator. **Wildfire year (2021):** the satellite becomes
competitive-to-better (its AOD retrieval sees the smoke plume's structure); its only
significant wins are in the Cascades — the sparsest, most heterogeneous network,
which *no* method reconstructs well. Script: `compare_acag.py`; paper App F.

## Episode-scale wildfire check (per-day)

`run_daily_wildfire.py` treats each day as an independent sparse field — per-day
spatial LOOCV, no temporal information. Northern Rockies 2021 (365 days, 41
monitors): elevation-detrending is neutral on clean days (+1.6%), **harms on smoke
days** (−12% median at 12–35 µg/m³, −20% above 35; paired Wilcoxon over smoke days
p<1e-4), and the elevation–PM correlation flattens as smoke arrives. This is the
daily-resolution confirmation of the annual wildfire counter-inference. Cascades and
Sierra runs pending the full daily download.

## Honest limitations

- PM2.5 fields are only weakly spatially structured (R² often 0.1–0.3), so NNI's
  edge over IDW is small and sometimes marginal; the value of the air-quality work
  is the elevation/counter-inference finding, not a large interpolation win.
- In some wildfire regions the annual field is genuinely patchy (negative R²) — a
  regime where *no* interpolator does well.
- The very-high-elevation bins are thin (few stations); the strong claim is the
  flat 0–1500 m range plus the wildfire reversal.
- "Truth" is the held-out station value; real networks cluster, which should
  favour NNI more than the near-uniform sampling used here.

## Usage

```bash
python -m experiments.run_temperature_benchmark --region Alps
python -m experiments.run_precip_benchmark     --region PacNW
python -m experiments.run_aq_benchmark         --region SouthernRockies --year 2021
python -m experiments.run_method_comparison    --dataset pm25 --region SouthernRockies
python -m experiments.run_timing
python -m experiments.compare_acag --nc "data/Annual 2/V6GL02.04.CNNPM25.NA.202301-202312.nc" --year 2023
python -m experiments.run_daily_wildfire --region NorthernRockies --year 2021
```

EPA AQS needs a free key in `.aqs.env` (`AQS_EMAIL`, `AQS_KEY`).

## Layout

```
aqv/                      the library
  interpolators.py        IDW + discrete-Sibson NNI (2-D, 3-D tiles, residual)
  validate.py             LOOCV, metrics, paired Wilcoxon, distance strata
  variogram.py            directional variogram, anisotropy factor (beta)
  data.py                 azimuthal-equidistant projection (lat/lon -> metres)
  baselines.py            GP/kriging, RF, GBM, MLP (same (x,y,z) inputs)
  loaders/                GHCN-Monthly, GHCN-Daily, EPA AQS, Copernicus DEM
experiments/              one module per experiment, run with python -m
  run_temperature_benchmark.py / run_precip_benchmark.py / run_aq_benchmark.py
  run_method_comparison.py    9-method stats-vs-ML bake-off (the 27 runs)
  run_timing.py               operational cost vs network size
  run_daily_wildfire.py       per-day episode-scale wildfire analysis
  compare_acag.py             satellite head-to-head
figures/                  figure generation (writes PDF + PNG)
scripts/                  sweep drivers, EPA daily pull
results/                  run outputs, not committed (see below)
data/                     datasets, gitignored, see data/README.md
```

## Reproducing

```bash
pip install -r requirements.txt
cp .aqs.env.example .aqs.env      # add your free EPA AQS key
# GHCN + ACAG files are manual downloads; see data/README.md

# everything runs as a module, from the repo root
python -m experiments.run_method_comparison --dataset temp --region Colorado
python -m experiments.run_timing
python -m figures.make_density_figure
bash scripts/run_all_bakeoffs.sh                       # all 27 benchmarks
```

Result files are not committed. Each runner writes its own tables and
per-station predictions under `results/`, so the numbers in the paper are
reproduced by running the batteries above (expect hours of LOOCV for the full
sweep, and an EPA AQS key for the air-quality regions). If you would rather not
re-run them, open an issue and I will share the generated tables and the
per-station predictions directly.

**Companion study.** The spatio-temporal GNN work (paper 2) lives in its own
repository; this repo is static spatial interpolation only.

## Licence

Code in this repository is MIT licensed (see `LICENSE`). The datasets are not
ours and keep their own terms: GHCN-Monthly and GHCN-Daily (NOAA) and EPA AQS are
US-government public domain, the Copernicus GLO-90 DEM and the ACAG V6.GL PM2.5
product each carry their own licence. None of them are redistributed here; the
loaders fetch them (see `data/README.md`).
