# Datasets

**Nothing in this directory is committed** (see `.gitignore`) — it is ~3.5 GB of
public data, all of it re-fetchable by the loaders in `aqv/`. This file documents
what goes where so a fresh clone can rebuild it.

| path | contents | source | fetched by |
|---|---|---|---|
| `ghcn/` | GHCN-Monthly v4 TAVG (`.dat` + `.inv`) | NOAA GHCNm | manual download, unpack here |
| `ghcnd/` | GHCN-Daily `2023.csv.gz` + `ghcnd-stations.txt` | NOAA GHCNd by_year | manual download |
| `epa_aqs/` | annual-mean PM2.5/PM10 JSON, one file per state-param-year | EPA AQS API | `aqv/epa_aqs.py` (auto-cached) |
| `epa_aqs_daily/` | daily PM2.5 JSON per state-year | EPA AQS API | `pull_aqs_daily.py` |
| `elevation_cache.json` | station elevations (lat,lon → m) | Copernicus GLO-90 via Open-Meteo | `aqv/elevation.py` (auto-cached) |
| `Annual 2/` | ACAG V6.GL.02.04 CNN PM2.5, native 0.01° North America | ACAG (WashU) | manual download |
| `Annual/` | same product at 0.1° (resolution-robustness check only) | ACAG | manual download |
| `cpcb/` | CPCB India single-hour snapshot (legacy, superseded) | data.gov.in | historical |
| `raw/` | early CPCB extracts (legacy) | — | historical |

## Credentials

EPA AQS needs a free key. Copy `.aqs.env.example` to `.aqs.env` in the repo root:

```
AQS_EMAIL=you@example.com
AQS_KEY=your_key_here
```

`.aqs.env` is gitignored. The loaders and `pull_aqs_daily.py` read it automatically.
Cached AQS JSONs echo the request URL, so **API keys can end up inside cached
responses** — this is why `data/` is excluded wholesale rather than by extension.

## Reproducing

Station elevations and AQS pulls are cached on first use, so re-runs are offline.
The ACAG and GHCN files must be downloaded manually (they have no API); the loaders
raise a clear `FileNotFoundError` naming the expected path if one is missing.
