"""Load GHCN-Monthly (GHCNm v4) TAVG station temperatures with elevation.

Real weather-station observations: sparse, mountainous coverage available, and
crucially each station carries an ELEVATION in its inventory metadata -- the
ideal test for whether elevation-aware interpolation helps for a variable
(temperature) that genuinely depends on altitude via the lapse rate, in contrast
to PM2.5 where it did not.

.inv  fixed width: ID(1-11) LAT(13-20) LON(22-30) ELEV(32-37) NAME(39-)
.dat  fixed width: ID(1-11) YEAR(12-15) 'TAVG'(16-19) then 12 x [VALUE(5) 3 flags]
      VALUE is in hundredths of degrees C; -9999 = missing.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def _find(root, ext):
    hits = glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no {ext} under {root}")
    return hits[0]


def load_inventory(root="data/ghcn"):
    """Return {station_id: (lat, lon, elev_m, name)}."""
    inv = {}
    with open(_find(root, ".inv")) as f:
        for ln in f:
            sid = ln[0:11]
            lat = float(ln[12:20]); lon = float(ln[21:30]); elev = float(ln[31:37])
            inv[sid] = (lat, lon, elev, ln[38:].strip())
    return inv


def load_year_annual_mean(year, root="data/ghcn", min_months=8):
    """Annual-mean TAVG (deg C) per station for ``year``.

    Requires at least ``min_months`` valid monthly values. Returns
    {station_id: mean_temp_C}.
    """
    tag = f"{year:4d}"
    out = {}
    with open(_find(root, ".dat")) as f:
        for ln in f:
            if ln[11:15] != tag or ln[15:19] != "TAVG":
                continue
            vals = []
            for m in range(12):
                s = 19 + m * 8
                v = int(ln[s:s + 5])
                if v != -9999:
                    vals.append(v / 100.0)
            if len(vals) >= min_months:
                out[ln[0:11]] = float(np.mean(vals))
    return out


def region_dataset(year, bbox, root="data/ghcn", min_months=8):
    """Build arrays for stations in ``bbox`` = (lat_min,lat_max,lon_min,lon_max).

    Returns dict with lat, lon, elev, temp (annual-mean deg C) over stations that
    have both a valid annual mean and inventory metadata inside the box.
    """
    inv = load_inventory(root)
    temps = load_year_annual_mean(year, root, min_months)
    la, lo, el, tp = [], [], [], []
    lat0, lat1, lon0, lon1 = bbox
    for sid, t in temps.items():
        if sid not in inv:
            continue
        lat, lon, elev, _ = inv[sid]
        if lat0 <= lat <= lat1 and lon0 <= lon <= lon1 and -999 < elev < 9000:
            la.append(lat); lo.append(lon); el.append(elev); tp.append(t)
    return {"lat": np.array(la), "lon": np.array(lo),
            "elev": np.array(el), "temp": np.array(tp), "n": len(la)}
