"""EPA AQS loader: annual-mean PM2.5 for a set of US states, with monitor
coordinates. Elevation is joined separately via aqv.elevation (Open-Meteo),
keeping the pipeline identical to the GHCN temperature path.

Requires a free EPA AQS API key (https://aqs.epa.gov/data/api/signup?email=...).
Pass email/key explicitly or via the AQS_EMAIL / AQS_KEY environment variables.

param codes: 88101 = PM2.5 FRM/FEM mass (24h); 88502 = PM2.5 non-FRM.
Western-mountain state FIPS: CO=08 UT=49 WY=56 NM=35 (AZ=04 ID=16 MT=30 NV=32).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

BASE = "https://aqs.epa.gov/data/api/annualData/byState"
CACHE_DIR = os.path.join("data", "epa_aqs")
MOUNTAIN_WEST = ["08", "49", "56", "35"]   # CO, UT, WY, NM
EPA_REGIONS = {
    "SouthernRockies": ["08", "49", "35", "04"],   # CO UT NM AZ
    "NorthernRockies": ["56", "30", "16"],         # WY MT ID
    "SierraGreatBasin": ["06", "32"],              # CA NV
    "Cascades": ["53", "41"],                      # WA OR
}
PM25_PARAMS = ["88101", "88502"]
PM10_PARAMS = ["81102"]


def _cache_path(state, param, year):
    return os.path.join(CACHE_DIR, f"annual_{state}_{param}_{year}.json")


def _fetch_state(email, key, state, param, year, retries=4, pause=3.0):
    """Raw annualData/byState pull for one state/param/year, cached to disk."""
    cp = _cache_path(state, param, year)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    q = urllib.parse.urlencode({
        "email": email, "key": key, "param": param, "state": state,
        "bdate": f"{year}0101", "edate": f"{year}1231",
    })
    url = f"{BASE}?{q}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                payload = json.load(r)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cp, "w") as f:
                json.dump(payload, f)
            return payload
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(pause * (attempt + 1))


def region_dataset(year, states=None, dataset="pm25", params=None, email=None, key=None,
                   pollutant_standard="auto", event_type="No Events"):
    """Return (lat, lon, val) monitor-level annual means over the given states.

    One monitor reports the same annual arithmetic mean under six pollutant
    standards and several durations/event-types; we pin a single standard and
    the No-Events series to avoid multiple-counting, then average any remaining
    rows (different POCs) per (lat, lon) site.
    """
    email = email or os.environ.get("AQS_EMAIL")
    key = key or os.environ.get("AQS_KEY")
    if not email or not key:
        raise RuntimeError("EPA AQS needs email+key (arg or AQS_EMAIL/AQS_KEY).")
    states = states or MOUNTAIN_WEST
    
    if params is None:
        if dataset == "pm10":
            params = PM10_PARAMS
            if pollutant_standard == "auto":
                pollutant_standard = None  # don't filter for PM10 standard yet
        else:
            params = PM25_PARAMS
            if pollutant_standard == "auto":
                pollutant_standard = "PM25 Annual 2012"

    # Collect annual means per site, bucketed by event_type. A monitor that
    # flagged exceptional events (e.g. wildfire smoke) in this year has NO
    # "No Events" row -- it reports only "Events Included/Excluded". Filtering to
    # "No Events" alone silently drops exactly the smoke-affected monitors,
    # biasing the wildfire-year field low. We instead PREFER "No Events" per site
    # and fall back to "Events Included" (the real, smoke-inclusive annual mean).
    prefer = []
    for e in (event_type, "No Events", "Events Included", "Events Excluded"):
        if e and e not in prefer:
            prefer.append(e)

    by_site = {}   # (rlat, rlon) -> {event_type: [annual arithmetic means]}
    for st in states:
        for pm in params:
            payload = _fetch_state(email, key, st, pm, year)
            for row in payload.get("Data", []):
                if pollutant_standard and row.get("pollutant_standard") != pollutant_standard:
                    continue
                et = row.get("event_type")
                val = row.get("arithmetic_mean")
                lat, lon = row.get("latitude"), row.get("longitude")
                if val is None or lat is None or lon is None:
                    continue
                k = (round(float(lat), 5), round(float(lon), 5))
                by_site.setdefault(k, {}).setdefault(et, []).append(float(val))

    def _pick(et_map):
        for e in prefer:                       # first available in preference order
            if et_map.get(e):
                return et_map[e]
        return [v for vs in et_map.values() for v in vs]   # else pool whatever exists

    lats, lons, vals = [], [], []
    for (la, lo), et_map in by_site.items():
        vs = _pick(et_map)
        if not vs:
            continue
        lats.append(la); lons.append(lo); vals.append(float(np.mean(vs)))
    order = np.argsort(lats)
    return (np.array(lats)[order], np.array(lons)[order], np.array(vals)[order])
