"""Precipitation benchmark (GHCN-Daily, mountain regions): the orographic middle
case. Precip-elevation is directional (windward up, rain-shadow down), so a linear
detrend should FAIL while horizontal NNI and the 3-D embedding may still help --
the interesting test of 'when/how to factor in height'.
"""
from __future__ import annotations
import argparse, functools
import numpy as np
from aqv.loaders.precip import region_dataset
from aqv.data import project_aeqd
from aqv.interpolators import idw_predict, nni_predict, nni_predict_3d, nni_predict_resid
from aqv.validate import metrics, paired_test, stratify_by_distance

BOX={"Colorado":(36,41,-109,-102),"WesternUS":(33,45,-125,-104),"Alps":(43,48,4,16),
     "Himalaya":(27,37,72,92),"SierraNevada":(35,42,-122,-117),"PacNW":(42,49,-124,-118),
     "Cascades":(45,49,-123,-120)}

def loo(xy, val, z, beta, predict):
    n=len(val); out=np.empty(n); alli=np.arange(n)
    for i in range(n):
        m=alli!=i
        out[i]=float(predict(xy[m], val[m], z[m], xy[i:i+1], z[i:i+1], beta))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023); ap.add_argument("--region", default="PacNW")
    ap.add_argument("--max_n", type=int, default=400)
    a=ap.parse_args()
    d=region_dataset(a.year, BOX[a.region])
    lat,lon,z,val=d["lat"],d["lon"],d["elev"],d["precip"]
    if len(val)>a.max_n:                         # keep it 'sparse': subsample
        rng=np.random.default_rng(0); s=rng.choice(len(val),a.max_n,replace=False)
        lat,lon,z,val=lat[s],lon[s],z[s],val[s]
    x,y,_=project_aeqd(lat,lon); xy=np.column_stack([x,y])
    slope=np.polyfit(z,val,1)[0]*1000
    print(f"{a.region} precip {a.year}: n={len(val)}  elev {z.min():.0f}-{z.max():.0f}m  "
          f"precip {val.min():.0f}-{val.max():.0f}mm")
    print(f"  precip-elev slope={slope:+.0f} mm/km  precip-elev r={np.corrcoef(z,val)[0,1]:+.2f} "
          f"(orographic: near-0 = windward/leeward cancel)\n")

    best=cfg=None
    for p in [1,2,3]:
        for k in [5,8,12]:
            pr=loo(xy,val,z,0,lambda tx,tv,tz,qx,qz,b,p=p,k=k: idw_predict(tx,tv,qx,power=p,k=k)[0])
            rr=metrics(val,pr)["rmse"]
            if best is None or rr<best: best,cfg=rr,(p,k)
    p,k=cfg; fb=functools.partial(idw_predict,power=p,k=k); idw_name="IDW(%d,%d)"%(p,k)
    methods={
      idw_name:    lambda tx,tv,tz,qx,qz,b: idw_predict(tx,tv,qx,power=p,k=k)[0],
      "NNI-2D":    lambda tx,tv,tz,qx,qz,b: nni_predict(tx,tv,qx,fallback=fb)[0][0],
      "NNI-3D":    lambda tx,tv,tz,qx,qz,b: nni_predict_3d(tx,tv,tz,qx,qz,25.0,fallback=fb)[0][0],
      "NNI-resid": lambda tx,tv,tz,qx,qz,b: nni_predict_resid(tx,tv,tz,qx,qz,degree=1,fallback=fb)[0][0],
    }
    preds={name:loo(xy,val,z,0,fn) for name,fn in methods.items()}
    base=metrics(val,preds[idw_name]); n2r=metrics(val,preds["NNI-2D"])["rmse"]
    print(f"{'method':12}{'RMSE':>9}{'MAE':>9}{'R2':>8}{'vs IDW':>9}{'vs NNI2D':>10}{'p(vsIDW)':>10}")
    print("-"*67)
    for name,pred in preds.items():
        mm=metrics(val,pred)
        dv=(base["rmse"]-mm["rmse"])/base["rmse"]*100; d2=(n2r-mm["rmse"])/n2r*100
        _,_,_,pv=paired_test(val,pred,preds[idw_name])
        print(f"{name:12}{mm['rmse']:>9.1f}{mm['mae']:>9.1f}{mm['r2']:>8.3f}{dv:>+8.1f}%{d2:>+9.1f}%{pv:>10.4f}")

if __name__=="__main__": main()
