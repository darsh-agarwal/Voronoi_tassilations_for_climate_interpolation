"""Full method bake-off on real sparse stations, identical LOOCV for all.

Tests the thesis: on sparse spatial data, do statistical / geostatistical methods
match or beat ML/DL given the SAME (x, y, elevation) information?

Families:
  geostat : IDW, NNI-2D (Voronoi), NNI-3D (3D tiles), NNI-resid (elevation covariate)
  kernel  : GP / kriging (2D and +elevation)   [GP regression == kriging]
  tree-ML : RandomForest, GradientBoosting (+elevation)
  DL      : MLP (+elevation)
"""
from __future__ import annotations
import argparse, functools, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from aqv.loaders.ghcn import region_dataset
from aqv.data import project_aeqd
from aqv.interpolators import idw_predict, nni_predict, nni_predict_3d, nni_predict_resid
from aqv.baselines import gp_predict, rf_predict, gbm_predict, mlp_predict
from aqv.variogram import anisotropy_factor
from aqv.validate import metrics, paired_test

BOX={"Colorado":(36,41,-109,-102),"WesternUS":(33,45,-125,-104),"Alps":(43,48,4,16),"Himalaya":(27,37,72,92),"Andes":(-40,-15,-78,-62),"Caucasus":(38,44,40,50),"SierraNevada":(35,42,-122,-117),"PacNW":(42,49,-124,-118)}

EPA_REGIONS={"SouthernRockies":["08","49","35","04"],"NorthernRockies":["56","30","16"],
             "SierraGreatBasin":["06","32"],"Cascades":["53","41"]}

def load_data(dataset, region, year):
    """Return (lat, lon, elev, value, unit, label). temp=GHCN, pm25=EPA AQS."""
    if dataset=="temp":
        d=region_dataset(year, BOX[region])
        return d["lat"],d["lon"],d["elev"],d["temp"],"C",f"{region} temp {year}"
    if dataset=="precip":
        from aqv.loaders.precip import region_dataset as precip_region
        d=precip_region(year, BOX[region])
        return d["lat"],d["lon"],d["elev"],d["precip"],"mm",f"{region} precip {year}"
    from aqv.loaders.epa_aqs import region_dataset as epa_region
    from aqv.loaders.elevation import get_elevation
    lat,lon,val=epa_region(year, states=EPA_REGIONS[region], dataset=dataset)
    z=get_elevation(lat,lon); ok=~np.isnan(z)
    label_str = "PM2.5" if dataset == "pm25" else "PM10"
    return lat[ok],lon[ok],z[ok],val[ok],"ug/m3",f"{region} {label_str} {year}"

def loocv(n, predict):
    out=np.empty(n); idx=np.arange(n)
    for i in range(n):
        out[i]=float(np.ravel(predict(idx!=i, i))[0])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", default="temp", choices=["temp","pm25","precip","pm10"])
    ap.add_argument("--region", default="Colorado"); ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--max_n", type=int, default=400)
    a=ap.parse_args()
    year=a.year or (2024 if a.dataset=="temp" else 2023)
    lat,lon,z,val,unit,label=load_data(a.dataset, a.region, year)
    if len(val)>a.max_n:
        rng0=np.random.default_rng(0); sub=rng0.choice(len(val),a.max_n,replace=False)
        lat,lon,z,val=lat[sub],lon[sub],z[sub],val[sub]
    x,y,_=project_aeqd(lat,lon); xy=np.column_stack([x,y])
    X3=np.column_stack([x,y,z]); n=len(val)
    beta,_=anisotropy_factor(xy,z,val)  # whole-set beta: display only (per-fold beta used below)
    print(f"{label}: n={n}  elev {z.min():.0f}-{z.max():.0f}m  {unit} {val.min():.1f}-{val.max():.1f}  beta={beta:.0f}\n")

    # --- nested per-fold tuning (NO leakage) ---------------------------------
    # For each held-out station i, IDW's (p,k) and the 3-D anisotropy beta are
    # fit on the TRAINING stations of that fold ONLY (via an inner LOOCV over
    # idx!=i). The held-out station never informs the baseline's settings, so the
    # earlier full-set-tuning leak is eliminated.
    from collections import Counter
    idx=np.arange(n); GRIDpk=[(pp,kk) for pp in (1,2,3) for kk in (5,8,12)]
    fold_pk=[None]*n; fold_beta=np.empty(n)
    for i in range(n):
        tr=idx!=i; xtr,vtr,ztr=xy[tr],val[tr],z[tr]; m=len(vtr); jdx=np.arange(m)
        best=cfg=None
        for pp,kk in GRIDpk:
            pr=np.array([idw_predict(xtr[jdx!=j],vtr[jdx!=j],xtr[j:j+1],power=pp,k=kk)[0]
                         for j in range(m)])
            r=metrics(vtr,pr)["rmse"]
            if best is None or r<best: best,cfg=r,(pp,kk)
        fold_pk[i]=cfg
        fold_beta[i]=anisotropy_factor(xtr,ztr,vtr)[0]
    fold_fb=[functools.partial(idw_predict,power=pk[0],k=pk[1]) for pk in fold_pk]
    pmode,kmode=Counter(fold_pk).most_common(1)[0][0]   # modal (p,k), for a stable label only

    methods=[
      ("IDW(%d,%d)"%(pmode,kmode),"geostat","xy", lambda m,i: idw_predict(xy[m],val[m],xy[i:i+1],power=fold_pk[i][0],k=fold_pk[i][1])),
      ("NNI-2D",          "geostat","xy",   lambda m,i: nni_predict(xy[m],val[m],xy[i:i+1],fallback=fold_fb[i])[0]),
      ("GP/kriging-2D",   "kernel", "xy",   lambda m,i: gp_predict(xy[m],val[m],xy[i:i+1])),
      ("NNI-3D tiles",    "geostat","xyz",  lambda m,i: nni_predict_3d(xy[m],val[m],z[m],xy[i:i+1],z[i:i+1],fold_beta[i],fallback=fold_fb[i])[0]),
      ("NNI-resid(elev)", "geostat","xyz",  lambda m,i: nni_predict_resid(xy[m],val[m],z[m],xy[i:i+1],z[i:i+1],degree=1,fallback=fold_fb[i])[0]),
      ("GP/kriging+elev", "kernel", "xyz",  lambda m,i: gp_predict(X3[m],val[m],X3[i:i+1])),
      ("RandomForest",    "tree-ML","xyz",  lambda m,i: rf_predict(X3[m],val[m],X3[i:i+1])),
      ("GradBoosting",    "tree-ML","xyz",  lambda m,i: gbm_predict(X3[m],val[m],X3[i:i+1])),
      ("MLP (DL)",        "DL",     "xyz",  lambda m,i: mlp_predict(X3[m],val[m],X3[i:i+1])),
    ]
    idw_pred=None; rows=[]; predmap={}
    for name,fam,feat,fn in methods:
        t=time.time(); pred=loocv(n,fn); mm=metrics(val,pred)
        predmap[name]=pred
        if name.startswith("IDW"): idw_pred=pred
        _,_,_,pv=paired_test(val,pred,idw_pred)
        rows.append((name,fam,feat,mm["rmse"],mm["mae"],mm["r2"],pv,time.time()-t))
        print(f"  ran {name:16} rmse={mm['rmse']:.3f}  ({time.time()-t:.0f}s)")

    print(f"\n{'method':17}{'family':9}{'feat':5}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'vsIDW%':>8}{'p':>9}")
    print("-"*72)
    base=[r for r in rows if r[0].startswith("IDW")][0][3]
    for name,fam,feat,rmse,mae,r2,pv,_ in sorted(rows,key=lambda r:r[3]):
        dv=(base-rmse)/base*100
        print(f"{name:17}{fam:9}{feat:5}{rmse:>8.3f}{mae:>8.3f}{r2:>8.3f}{dv:>+7.1f}%{pv:>9.4f}")

    # head-to-head: is the best ML/DL actually distinguishable from the best geostat?
    geo=min([r for r in rows if r[1]=="geostat" and r[2]=="xyz"], key=lambda r:r[3])[0]
    print(f"\nHead-to-head vs best geostat ({geo}, rmse={metrics(val,predmap[geo])['rmse']:.3f}):")
    for name,fam,feat,rmse,*_ in sorted(rows,key=lambda r:r[3]):
        if fam in ("kernel","tree-ML","DL") and feat=="xyz":
            _,_,_,pv=paired_test(val,predmap[name],predmap[geo])
            gap=(metrics(val,predmap[geo])['rmse']-rmse)
            verdict="ML better" if gap>0 else "geostat better"
            sig="significant" if pv<0.05 else "NOT significant (indistinguishable)"
            print(f"  {name:16} delta_rmse={gap:+.3f}  p={pv:.4f}  -> {verdict}, {sig}")

    # optional machine-readable dump (set BAKEOFF_CSV to a path to append rows)
    import os, csv as _csv
    csvpath=os.environ.get("BAKEOFF_CSV")
    if csvpath:
        base_rmse=[r for r in rows if r[0].startswith("IDW")][0][3]
        newfile=not os.path.exists(csvpath)
        with open(csvpath,"a",newline="") as f:
            w=_csv.writer(f)
            if newfile:
                w.writerow(["variable","region","year","n","method","family","feat",
                            "rmse","mae","r2","vs_idw_pct","p_vs_idw"])
            for name,fam,feat,rmse,mae,r2,pv,_ in rows:
                vs=(base_rmse-rmse)/base_rmse*100
                w.writerow([a.dataset,a.region,year,n,name,fam,feat,
                            f"{rmse:.3f}",f"{mae:.3f}",f"{r2:.3f}",f"{vs:+.1f}",f"{pv:.4f}"])

    # optional per-station prediction dump (set PRED_DIR) for later equivalence tests
    preddir=os.environ.get("PRED_DIR")
    if preddir:
        os.makedirs(preddir, exist_ok=True)
        safe={k.replace("/","_").replace(" ","_").replace("(","").replace(")",""):v
              for k,v in predmap.items()}
        np.savez(os.path.join(preddir,f"{a.dataset}_{a.region}_{year}.npz"), val=val, **safe)

if __name__=="__main__": main()
