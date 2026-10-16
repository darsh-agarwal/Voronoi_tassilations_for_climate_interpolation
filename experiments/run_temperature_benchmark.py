"""Temperature benchmark (GHCN, mountainous region): does elevation help here?

Elevation-aware battery (IDW, NNI-2D, NNI-3D, NNI-residual) for annual-mean temperature on real
sparse weather stations. Temperature depends on elevation via the lapse rate, so
-- unlike annual-mean PM2.5 -- elevation-aware interpolation is expected to WIN.
Leave-one-station-out CV; tuned IDW baseline; paired Wilcoxon significance.
"""
from __future__ import annotations
import argparse, functools
import numpy as np
from aqv.loaders.ghcn import region_dataset
from aqv.data import project_aeqd
from aqv.interpolators import idw_predict, nni_predict, nni_predict_3d, nni_predict_resid
from aqv.variogram import anisotropy_factor
from aqv.validate import metrics, paired_test

BOX={"Colorado":(36,41,-109,-102),"WesternUS":(33,45,-125,-104),"Alps":(43,48,4,16),"Himalaya":(27,37,72,92),"Andes":(-40,-15,-78,-62),"Caucasus":(38,44,40,50),"SierraNevada":(35,42,-122,-117),"PacNW":(42,49,-124,-118)}

def loo(xy, val, z, beta, predict):
    """predict(txy,tval,tz,qxy,qz,beta) -> scalar; leave-one-out over stations."""
    n=len(val); out=np.empty(n); alli=np.arange(n)
    for i in range(n):
        m=alli!=i
        out[i]=float(predict(xy[m], val[m], z[m], xy[i:i+1], z[i:i+1], beta))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--region", default="Colorado")
    a=ap.parse_args()
    d=region_dataset(a.year, BOX[a.region])
    x,y,_=project_aeqd(d["lat"],d["lon"]); xy=np.column_stack([x,y])
    val=d["temp"]; z=d["elev"]
    slope=np.polyfit(z,val,1)[0]*1000
    beta,info=anisotropy_factor(xy,z,val)
    av=info.get('a_v'); av=f"{av:.0f}m" if av else "n/a"
    print(f"{a.region} {a.year}: n={d['n']}  elev {z.min():.0f}-{z.max():.0f}m  T {val.min():.1f}-{val.max():.1f}C")
    print(f"  lapse rate={slope:.2f} C/km  elev-T r={np.corrcoef(z,val)[0,1]:+.2f}  "
          f"beta={beta:.1f} (a_h={info.get('a_h'):.0f}m a_v={av})\n")

    # tune IDW by LOOCV RMSE
    best=cfg=None
    for p in [1,2,3]:
        for k in [5,8,12]:
            pr=loo(xy,val,z,beta,lambda tx,tv,tz,qx,qz,b,p=p,k=k: idw_predict(tx,tv,qx,power=p,k=k)[0])
            rr=metrics(val,pr)["rmse"]
            if best is None or rr<best: best,cfg=rr,(p,k)
    p,k=cfg; fb=functools.partial(idw_predict,power=p,k=k)

    methods={
      "IDW(%d,%d)"%(p,k): lambda tx,tv,tz,qx,qz,b: idw_predict(tx,tv,qx,power=p,k=k)[0],
      "NNI-2D":    lambda tx,tv,tz,qx,qz,b: nni_predict(tx,tv,qx,fallback=fb)[0][0],
      "NNI-3D":    lambda tx,tv,tz,qx,qz,b: nni_predict_3d(tx,tv,tz,qx,qz,b,fallback=fb)[0][0],
      "NNI-resid": lambda tx,tv,tz,qx,qz,b: nni_predict_resid(tx,tv,tz,qx,qz,degree=1,fallback=fb)[0][0],
    }
    preds={name:loo(xy,val,z,beta,fn) for name,fn in methods.items()}
    base=metrics(val,preds["IDW(%d,%d)"%(p,k)]); n2r=metrics(val,preds["NNI-2D"])["rmse"]
    print(f"{'method':12}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'vs IDW':>9}{'vs NNI2D':>10}{'p(vsIDW)':>10}")
    print("-"*65)
    for name,pred in preds.items():
        mm=metrics(val,pred)
        dv=(base["rmse"]-mm["rmse"])/base["rmse"]*100
        d2=(n2r-mm["rmse"])/n2r*100
        _,_,_,pv=paired_test(val,pred,preds["IDW(%d,%d)"%(p,k)])
        print(f"{name:12}{mm['rmse']:>8.3f}{mm['mae']:>8.3f}{mm['r2']:>8.3f}{dv:>+8.1f}%{d2:>+9.1f}%{pv:>10.4f}")

if __name__=="__main__": main()
