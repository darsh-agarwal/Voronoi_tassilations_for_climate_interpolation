"""Real-station PM2.5 benchmark (EPA AQS, mountain-west): does elevation help?

The honest counterpart to the temperature study. Unlike the ACAG satellite
product (a smooth CNN model output that already bakes in topography), these are
REAL sparse monitors -- so whatever elevation does here is physics, not a
dataset artifact. Same battery as run_temperature_benchmark.py: tuned-IDW
baseline, NNI-2D/3D/residual, variogram-beta diagnostic, paired Wilcoxon, plus a
distance-to-nearest-station stratification (the under-monitored-area story).
"""
from __future__ import annotations
import argparse, functools
import numpy as np
from aqv.loaders.epa_aqs import region_dataset, MOUNTAIN_WEST, EPA_REGIONS
from aqv.loaders.elevation import get_elevation
from aqv.data import project_aeqd
from aqv.interpolators import idw_predict, nni_predict, nni_predict_3d, nni_predict_resid
from aqv.variogram import anisotropy_factor
from aqv.validate import metrics, paired_test, stratify_by_distance

def loo(xy, val, z, beta, predict):
    n=len(val); out=np.empty(n); alli=np.arange(n)
    for i in range(n):
        m=alli!=i
        out[i]=float(predict(xy[m], val[m], z[m], xy[i:i+1], z[i:i+1], beta))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--region", default="SouthernRockies")
    ap.add_argument("--dataset", default="pm25", choices=["pm25", "pm10"])
    a=ap.parse_args()
    states=EPA_REGIONS.get(a.region, MOUNTAIN_WEST)
    lat,lon,val=region_dataset(a.year, states=states, dataset=a.dataset)
    z=get_elevation(lat,lon)
    ok=~np.isnan(z)
    lat,lon,val,z=lat[ok],lon[ok],val[ok],z[ok]
    x,y,_=project_aeqd(lat,lon); xy=np.column_stack([x,y])
    slope=np.polyfit(z,val,1)[0]*1000
    beta,info=anisotropy_factor(xy,z,val)
    av=info.get('a_v'); av=f"{av:.0f}m" if av else "n/a"
    name_str = "PM2.5" if a.dataset == "pm25" else "PM10"
    print(f"EPA {a.region} {name_str} {a.year}: n={len(val)}  elev {z.min():.0f}-{z.max():.0f}m  "
          f"{name_str} {val.min():.1f}-{val.max():.1f} ug/m3")
    print(f"  {name_str}-elev slope={slope:.2f} (ug/m3)/km  elev-PM r={np.corrcoef(z,val)[0,1]:+.2f}  "
          f"beta={beta:.1f} (a_h={info.get('a_h'):.0f}m a_v={av})\n")

    best=cfg=None
    for p in [1,2,3]:
        for k in [5,8,12]:
            pr=loo(xy,val,z,beta,lambda tx,tv,tz,qx,qz,b,p=p,k=k: idw_predict(tx,tv,qx,power=p,k=k)[0])
            rr=metrics(val,pr)["rmse"]
            if best is None or rr<best: best,cfg=rr,(p,k)
    p,k=cfg; fb=functools.partial(idw_predict,power=p,k=k)
    idw_name="IDW(%d,%d)"%(p,k)

    methods={
      idw_name:    lambda tx,tv,tz,qx,qz,b: idw_predict(tx,tv,qx,power=p,k=k)[0],
      "NNI-2D":    lambda tx,tv,tz,qx,qz,b: nni_predict(tx,tv,qx,fallback=fb)[0][0],
      "NNI-3D":    lambda tx,tv,tz,qx,qz,b: nni_predict_3d(tx,tv,tz,qx,qz,b,fallback=fb)[0][0],
      "NNI-resid": lambda tx,tv,tz,qx,qz,b: nni_predict_resid(tx,tv,tz,qx,qz,degree=1,fallback=fb)[0][0],
    }
    preds={name:loo(xy,val,z,beta,fn) for name,fn in methods.items()}
    base=metrics(val,preds[idw_name]); n2r=metrics(val,preds["NNI-2D"])["rmse"]
    print(f"{'method':12}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'vs IDW':>9}{'vs NNI2D':>10}{'p(vsIDW)':>10}")
    print("-"*65)
    for name,pred in preds.items():
        mm=metrics(val,pred)
        dv=(base["rmse"]-mm["rmse"])/base["rmse"]*100
        d2=(n2r-mm["rmse"])/n2r*100
        _,_,_,pv=paired_test(val,pred,preds[idw_name])
        print(f"{name:12}{mm['rmse']:>8.3f}{mm['mae']:>8.3f}{mm['r2']:>8.3f}{dv:>+8.1f}%{d2:>+9.1f}%{pv:>10.4f}")

    print("\ndistance-to-nearest-station strata (RMSE per method; gain should widen when sparse):")
    rows=stratify_by_distance(val, preds, xy)
    cols=list(preds.keys())
    print(f"{'dist_km':>14}{'n':>5}"+"".join(f"{c:>11}"for c in cols))
    for r in rows:
        band=f"{r['dist_lo_km']}-{r['dist_hi_km']}"
        print(f"{band:>14}{r['n']:>5}"+"".join(f"{r[c]:>11.3f}"for c in cols))

if __name__=="__main__": main()
