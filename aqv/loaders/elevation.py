"""Elevation lookup for lat/lon points, via the Open-Meteo elevation API.

Open-Meteo serves the Copernicus GLO-90 DEM, needs no API key, and accepts
batched coordinates. Results are cached to disk (keyed by rounded lat/lon) so a
region is fetched once and re-runs are offline.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

CACHE = os.path.join("data", "elevation_cache.json")
BATCH = 100           # coords per request (URL-length safe for GET)
ROUND = 5             # cache key precision (~1 m at these latitudes)
URL = "https://api.open-meteo.com/v1/elevation"
INTER_BATCH_PAUSE = 2.0   # seconds between requests, to stay under the rate cap


def _load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f)


def _key(lat, lon):
    return f"{round(float(lat), ROUND)},{round(float(lon), ROUND)}"


def get_elevation(lat, lon, retries=5, pause=0.5):
    """Return an elevation array (metres) for parallel lat/lon arrays.

    Cache hits are served locally; misses are fetched in batches and cached.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    cache = _load_cache()
    out = np.full(len(lat), np.nan)

    misses = []
    for i in range(len(lat)):
        k = _key(lat[i], lon[i])
        if k in cache:
            out[i] = cache[k]
        else:
            misses.append(i)

    n_batches = (len(misses) + BATCH - 1) // BATCH
    for bi, b in enumerate(range(0, len(misses), BATCH)):
        idx = misses[b:b + BATCH]
        q = urllib.parse.urlencode({
            "latitude": ",".join(str(lat[i]) for i in idx),
            "longitude": ",".join(str(lon[i]) for i in idx),
        })
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(f"{URL}?{q}", timeout=30) as r:
                    elev = json.load(r)["elevation"]
                for j, i in enumerate(idx):
                    out[i] = elev[j]
                    cache[_key(lat[i], lon[i])] = float(elev[j])
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:            # rate limited -> long backoff
                    time.sleep(10 * (attempt + 1))
                elif attempt == retries - 1:
                    raise
                else:
                    time.sleep(pause * (attempt + 1))
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(pause * (attempt + 1))
        _save_cache(cache)                   # persist incrementally
        if bi < n_batches - 1:
            time.sleep(INTER_BATCH_PAUSE)

    _save_cache(cache)
    return out
