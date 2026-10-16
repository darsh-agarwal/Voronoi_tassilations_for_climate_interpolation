"""Regenerate the paper's satellite comparison (Table tab:acag) end-to-end.

Head-to-head: our interpolators vs the ACAG V6.GL CNN PM2.5 product, BOTH
scored against real held-out EPA annual-mean station values.

Unlike the earlier version of this script (which re-ran a single NNI-residual
LOOCV with its own IDW fallback tuning), this one loads the canonical nested-CV
per-station predictions from results/preds/pm25_{region}_{year}.npz -- the same
arrays behind every other table in the paper -- so the "ours" column is exactly
the paper's, with no leaky global IDW tuning and no drift between tables.

Per region-year it emits:
  n, CV-selected NNI variant (argmin LOOCV RMSE over NNI-2D / NNI-3D / residual),
  ours RMSE+R2, ACAG RMSE+R2, paired two-sided Wilcoxon p (selected vs ACAG),
  and `beat` = how many of the nine benchmarked methods out-RMSE the satellite.
It then prints the full per-method p-value table (the "significantly beats
several methods" evidence) for each run.

    python compare_acag.py                    # native 0.01 deg grids, both years
    python compare_acag.py --res 0p10         # 0.1 deg robustness check
    python compare_acag.py --years 2023       # one year only

Station alignment with the npz is asserted (np.allclose on the value vector),
so the script fails loudly rather than silently mis-pairing stations.
"""
from __future__ import annotations
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np

from aqv.loaders.epa_aqs import region_dataset as epa_region
from aqv.loaders.elevation import get_elevation
from aqv.validate import metrics, paired_test

EPA_REGIONS = {"SouthernRockies": ["08", "49", "35", "04"],
               "NorthernRockies": ["56", "30", "16"],
               "SierraGreatBasin": ["06", "32"], "Cascades": ["53", "41"]}
REGION_ORDER = ["SouthernRockies", "NorthernRockies", "SierraGreatBasin", "Cascades"]

NC_DIR = {"native": "data/Annual 2", "0p10": "data/Annual"}
NC_PAT = {"native": "V6GL02.04.CNNPM25.NA.{y}01-{y}12.nc",
          "0p10":   "V6GL02.04.CNNPM25.0p10.NA.{y}01-{y}12.nc"}
PREDS = "results/preds/pm25_{region}_{year}.npz"

NNI_VARIANTS = ["NNI-2D", "NNI-3D_tiles", "NNI-residelev"]
PRETTY = {"NNI-2D": "NNI-2D", "NNI-3D_tiles": "NNI-3D", "NNI-residelev": "NNI-res"}
NODATA = -900.0


def load_grid(nc_path):
    import netCDF4 as nc
    ds = nc.Dataset(nc_path)
    return ds.variables["lat"][:], ds.variables["lon"][:], ds.variables["PM25"]


def sample_grid(lat_g, lon_g, PM, qlat, qlon):
    """Nearest-pixel sample; NaN where out-of-grid or no-data. Handles 0..360
    longitude grids by wrapping negative query longitudes."""
    vals = np.full(len(qlat), np.nan)
    inside = np.zeros(len(qlat), bool)
    lo_lat, hi_lat = float(lat_g.min()), float(lat_g.max())
    lo_lon, hi_lon = float(lon_g.min()), float(lon_g.max())
    lon_360 = hi_lon > 180.0
    for k, (la, lo) in enumerate(zip(qlat, qlon)):
        qlo = lo % 360.0 if lon_360 else lo
        if not (lo_lat <= la <= hi_lat and lo_lon <= qlo <= hi_lon):
            continue
        i = int(np.abs(lat_g - la).argmin())
        j = int(np.abs(lon_g - qlo).argmin())
        v = float(np.ma.filled(PM[i, j], NODATA - 1))
        if v > NODATA:
            vals[k] = v; inside[k] = True
    return vals, inside


def stations(region, year):
    """EPA stations with elevation, in the same order/filtering the benchmark used."""
    lat, lon, val = epa_region(year, states=EPA_REGIONS[region], dataset="pm25")
    z = get_elevation(lat, lon)
    ok = ~np.isnan(z)
    return lat[ok], lon[ok], val[ok]


def run(year, res):
    nc_path = os.path.join(NC_DIR[res], NC_PAT[res].format(y=year))
    lat_g, lon_g, PM = load_grid(nc_path)
    rows, detail = [], []
    for region in REGION_ORDER:
        lat, lon, val = stations(region, year)
        acag, inside = sample_grid(lat_g, lon_g, PM, lat, lon)
        lat, lon, val, acag = lat[inside], lon[inside], val[inside], acag[inside]

        d = np.load(PREDS.format(region=region, year=year))
        if len(d["val"]) != len(val) or not np.allclose(np.sort(d["val"]), np.sort(val), atol=1e-9):
            raise SystemExit(f"{region} {year}: station set does not match "
                             f"{PREDS.format(region=region, year=year)} "
                             f"(npz n={len(d['val'])}, here n={len(val)}). "
                             "EPA/elevation caches out of sync with stored predictions.")
        if not np.allclose(d["val"], val, atol=1e-9):
            raise SystemExit(f"{region} {year}: station ORDER differs from npz; refusing to mis-pair.")

        methods = [k for k in d.keys() if k != "val"]
        rmse = {m: metrics(val, d[m])["rmse"] for m in methods}
        ma = metrics(val, acag)
        beat = sum(rmse[m] < ma["rmse"] for m in methods)

        sel = min(NNI_VARIANTS, key=lambda m: rmse[m])
        ms = metrics(val, d[sel])
        _, _, _, p_sel = paired_test(val, d[sel], acag)

        rows.append((region, len(val), PRETTY[sel], ms["rmse"], ma["rmse"],
                     ms["r2"], ma["r2"], p_sel, beat, len(methods)))
        detail.append((region, len(val),
                       sorted(((m, rmse[m], paired_test(val, d[m], acag)[3]) for m in methods),
                              key=lambda t: t[1]), ma["rmse"]))
    return rows, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", choices=("native", "0p10"), default="native",
                    help="ACAG grid: native 0.01 deg or the 0.1 deg robustness check")
    ap.add_argument("--years", nargs="*", type=int, default=[2023, 2021])
    a = ap.parse_args()

    print(f"ACAG V6.GL comparison -- grid: {a.res}"
          f" ({'0.01' if a.res == 'native' else '0.1'} deg)\n")
    for year in a.years:
        rows, detail = run(year, a.res)
        tag = "normal year" if year == 2023 else "wildfire year"
        print(f"== {year} ({tag}) ==")
        print(f"{'region':17}{'n':>4}  {'method':8}{'ours':>7}{'ACAG':>7}"
              f"{'our_R2':>8}{'ACAG_R2':>9}{'p':>8}{'beat':>7}")
        for r, n, sel, rn, ra, r2n, r2a, p, beat, nm in rows:
            print(f"{r:17}{n:>4}  {sel:8}{rn:>7.3f}{ra:>7.3f}"
                  f"{r2n:>8.2f}{r2a:>9.2f}{p:>8.3f}{beat:>5}/{nm}")
        print()
        for r, n, meths, ra in detail:
            print(f"-- {r} {year} (n={n}, ACAG RMSE {ra:.3f}); per-method RMSE / p vs ACAG:")
            for m, rm, p in meths:
                mark = "beats ACAG" if rm < ra else ("  p<0.05" if p < 0.05 else "")
                print(f"     {m:18}{rm:>8.3f}   p={p:.4f}  {mark}")
            print()


if __name__ == "__main__":
    main()
