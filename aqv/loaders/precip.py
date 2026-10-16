"""GHCN-Daily annual precipitation for a bounding box, with station elevation.

Precipitation is the second pillar of climate and -- unlike temperature's clean
lapse rate -- its elevation dependence is OROGRAPHIC (rises windward, collapses
in the rain shadow): the interesting middle case for 'should we factor in height'.

Reads the fixed-width station list (lat/lon/elev built in, no DEM needed) and the
by_year daily CSV, summing PRCP (tenths of mm) to an annual total (mm) per station.
"""
from __future__ import annotations
import gzip, os
import numpy as np

STATIONS = os.path.join("data", "ghcnd", "ghcnd-stations.txt")

def _stations_in_box(bbox):
    lat0,lat1,lon0,lon1 = bbox
    out={}
    with open(STATIONS) as f:
        for ln in f:
            try:
                lat=float(ln[12:20]); lon=float(ln[21:30]); elev=float(ln[31:37])
            except ValueError:
                continue
            if elev<=-999: continue
            if lat0<=lat<=lat1 and lon0<=lon<=lon1:
                out[ln[0:11]]=(lat,lon,elev)
    return out

def region_dataset(year, bbox, root="data/ghcnd", min_days=300):
    """Return dict lat/lon/elev/precip(mm annual)/n for stations in bbox.

    Only stations with >= min_days of quality-flag-clean PRCP are kept (annual
    total needs near-complete coverage to be comparable).
    """
    sites=_stations_in_box(bbox)
    ssum={}; scnt={}
    path=os.path.join(root, f"{year}.csv.gz")
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ",PRCP," not in ln: continue
            p=ln.split(",")
            sid=p[0]
            if sid not in sites: continue
            if p[5]!="": continue          # Q-FLAG non-blank => failed QC
            ssum[sid]=ssum.get(sid,0.0)+float(p[3])
            scnt[sid]=scnt.get(sid,0)+1
    la,lo,el,pr=[],[],[],[]
    for sid,(lat,lon,elev) in sites.items():
        if scnt.get(sid,0)>=min_days:
            la.append(lat); lo.append(lon); el.append(elev)
            pr.append(ssum[sid]/10.0)      # tenths of mm -> mm
    return {"lat":np.array(la),"lon":np.array(lo),"elev":np.array(el),
            "precip":np.array(pr),"n":len(la)}
