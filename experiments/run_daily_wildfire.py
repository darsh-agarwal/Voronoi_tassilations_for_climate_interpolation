"""Per-day spatial LOOCV through a wildfire season: does the elevation-PM
relationship flip sign during smoke episodes, and does elevation-detrending
harm on exactly those days?

Each day is treated as an independent sparse field (no temporal information is
shared between days -- purely spatial interpolation, day by day). For every day
with enough reporting monitors we run leave-one-station-out CV for tuned IDW,
NNI-2D and NNI-residual, and record the day's elevation-PM correlation and
regional mean PM2.5. Days are then binned by smoke intensity.

Data: EPA AQS dailyData (data/epa_aqs_daily/, via pull_aqs_daily.py), param
88101, pollutant_standard pinned to 'PM25 24-hour 2012' to undo the 8x standard
multiplicity, 24-hour durations only, validity Y. Rows flagged 'Events
Excluded' / 'Concurred Events Excluded' are dropped (we want the smoke-inclusive
measurement); per (site, day) any remaining POC/duration duplicates are averaged.

    python run_daily_wildfire.py --region NorthernRockies --year 2021
"""
from __future__ import annotations
import argparse, collections, functools, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np

from aqv.loaders.epa_aqs import EPA_REGIONS
from aqv.loaders.elevation import get_elevation
from aqv.data import project_aeqd
from aqv.interpolators import idw_predict, nni_predict, nni_predict_resid
from aqv.validate import metrics

DAILY_DIR = os.path.join("data", "epa_aqs_daily")
STANDARD = "PM25 24-hour 2012"
DUR_OK = {"24 HOUR", "24-HR BLK AVG"}
EV_DROP = {"Events Excluded", "Concurred Events Excluded"}


def load_daily(region, year, param="88101"):
    """Return {date: {(lat,lon): mean_pm}} pooled over the region's states."""
    days = collections.defaultdict(lambda: collections.defaultdict(list))
    for st in EPA_REGIONS[region]:
        path = os.path.join(DAILY_DIR, f"daily_{st}_{param}_{year}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing -- rerun pull_aqs_daily.py")
        payload = json.load(open(path))
        rows = payload["Data"] if isinstance(payload, dict) else payload
        for r in rows:
            if r.get("pollutant_standard") != STANDARD: continue
            if r.get("sample_duration") not in DUR_OK: continue
            if r.get("validity_indicator") not in (None, "Y"): continue
            if r.get("event_type") in EV_DROP: continue
            v, la, lo, d = (r.get("arithmetic_mean"), r.get("latitude"),
                            r.get("longitude"), r.get("date_local"))
            if v is None or la is None or lo is None or not d: continue
            days[d][(round(float(la), 5), round(float(lo), 5))].append(float(v))
    return {d: {s: float(np.mean(v)) for s, v in sites.items()}
            for d, sites in days.items()}


def tune_idw(xy, val):
    n = len(val); idx = np.arange(n); best = cfg = None
    for p in (1, 2, 3):
        for k in (5, 8, 12):
            pr = np.array([idw_predict(xy[idx != i], val[idx != i], xy[i:i+1],
                                       power=p, k=k)[0] for i in range(n)])
            r = float(np.sqrt(np.mean((pr - val) ** 2)))
            if best is None or r < best: best, cfg = r, (p, k)
    return cfg, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="NorthernRockies", choices=list(EPA_REGIONS))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--min_n", type=int, default=15)
    a = ap.parse_args()

    daily = load_daily(a.region, a.year)
    all_sites = sorted({s for d in daily.values() for s in d})
    lats = np.array([s[0] for s in all_sites]); lons = np.array([s[1] for s in all_sites])
    elev_all = get_elevation(lats, lons)
    elev_map = {s: e for s, e in zip(all_sites, elev_all) if np.isfinite(e)}
    print(f"{a.region} {a.year}: {len(daily)} days, {len(all_sites)} unique monitors "
          f"({len(elev_map)} with elevation)")

    rows = []
    for d in sorted(daily):
        sites = [(s, v) for s, v in daily[d].items() if s in elev_map]
        if len(sites) < a.min_n: continue
        lat = np.array([s[0][0] for s in sites]); lon = np.array([s[0][1] for s in sites])
        val = np.array([s[1] for s in sites]); z = np.array([elev_map[s[0]] for s in sites])
        x, y, _ = project_aeqd(lat, lon); xy = np.column_stack([x, y])
        n = len(val); idx = np.arange(n)
        (p, k), rmse_idw = tune_idw(xy, val)
        fb = functools.partial(idw_predict, power=p, k=k)
        pr2, prr = np.empty(n), np.empty(n)
        for i in range(n):
            m = idx != i
            pr2[i] = nni_predict(xy[m], val[m], xy[i:i+1], fallback=fb)[0][0]
            prr[i] = nni_predict_resid(xy[m], val[m], z[m], xy[i:i+1], z[i:i+1],
                                       degree=1, fallback=fb)[0][0]
        r_elev = float(np.corrcoef(z, val)[0, 1]) if np.ptp(z) > 0 else np.nan
        rows.append(dict(date=d, n=n, mean_pm=float(val.mean()),
                         max_pm=float(val.max()), r_elev=r_elev,
                         rmse_idw=rmse_idw,
                         rmse_nni2d=metrics(val, pr2)["rmse"],
                         rmse_resid=metrics(val, prr)["rmse"]))
        if len(rows) % 50 == 0: print(f"  {len(rows)} days done ({d})")

    os.makedirs("results", exist_ok=True)
    out = f"results/daily_wildfire_{a.region}_{a.year}.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} days with n>={a.min_n})")

    # summary by smoke tier (day's regional mean PM2.5)
    tiers = [("clean  <12", lambda m: m < 12), ("moderate 12-35", lambda m: 12 <= m < 35),
             ("smoke  >=35", lambda m: m >= 35)]
    print(f"\n{'tier':16}{'days':>6}{'med r(elev)':>12}{'IDW':>8}{'NNI2D':>8}{'resid':>8}{'resid vs 2D':>13}")
    for name, cond in tiers:
        sel = [r for r in rows if cond(r["mean_pm"])]
        if not sel: print(f"{name:16}{0:>6}"); continue
        med = lambda key: float(np.median([r[key] for r in sel]))
        dlt = 100 * (med("rmse_nni2d") - med("rmse_resid")) / med("rmse_nni2d")
        print(f"{name:16}{len(sel):>6}{med('r_elev'):>12.2f}{med('rmse_idw'):>8.2f}"
              f"{med('rmse_nni2d'):>8.2f}{med('rmse_resid'):>8.2f}{dlt:>+12.1f}%")

    from scipy import stats
    smoke = [r for r in rows if r["mean_pm"] >= 20]
    clean = [r for r in rows if r["mean_pm"] < 12]
    if len(smoke) >= 5:
        w = stats.wilcoxon([r["rmse_resid"] for r in smoke], [r["rmse_nni2d"] for r in smoke])
        print(f"\nsmoke days (mean>=20, n={len(smoke)}): resid vs 2D paired Wilcoxon p={w.pvalue:.4f}")
    if clean and smoke:
        print(f"median r(elev): clean {np.median([r['r_elev'] for r in clean]):+.2f}"
              f"  ->  smoke {np.median([r['r_elev'] for r in smoke]):+.2f}")

if __name__ == "__main__":
    main()
