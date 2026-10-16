"""Operational timing: when new observations arrive, how long to produce updated
predictions? Memory-based statistical interpolators (IDW/NNI) need NO refit;
ML/kriging must retrain (GP is O(n^3)). We time the full 'ingest n stations ->
predict Q targets' cycle as the network grows -- the real-time-at-scale argument.
Wall-clock, median of R repeats, on real GHCN temperature stations.
"""
from __future__ import annotations
import argparse, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from aqv.loaders.ghcn import region_dataset
from aqv.data import project_aeqd
from aqv.interpolators import idw_predict, nni_predict, nni_predict_3d
from aqv.baselines import gp_predict, rf_predict, gbm_predict, mlp_predict

BOX={"WesternUS":(33,45,-125,-104)}

def timeit(fn, R):
    ts=[]
    for _ in range(R):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=300); ap.add_argument("--R", type=int, default=3)
    a=ap.parse_args()
    d=region_dataset(2024, BOX["WesternUS"])
    x,y,_=project_aeqd(d["lat"],d["lon"]); z=d["elev"]; val=d["temp"]
    XY=np.column_stack([x,y]); XYZ=np.column_stack([x,y,z]); N=len(val)
    rng=np.random.default_rng(0)
    n_list=[n for n in [100,300,1000,N] if n<=N]
    # fixed query set (Q random in-region points)
    qi=rng.choice(N, size=min(a.Q,N), replace=False)
    qxy=XY[qi]; qz=z[qi]; qxyz=XYZ[qi]

    methods=[
      ("IDW",       "stats", lambda tr: idw_predict(XY[tr],val[tr],qxy,power=2,k=12)),
      ("NNI-2D",    "stats", lambda tr: nni_predict(XY[tr],val[tr],qxy)),
      ("NNI-3D",    "stats", lambda tr: nni_predict_3d(XY[tr],val[tr],z[tr],qxy,qz,25.0)),
      ("GP/kriging","ML",    lambda tr: gp_predict(XYZ[tr],val[tr],qxyz)),
      ("RandForest","ML",    lambda tr: rf_predict(XYZ[tr],val[tr],qxyz)),
      ("GradBoost", "ML",    lambda tr: gbm_predict(XYZ[tr],val[tr],qxyz)),
      ("MLP",       "ML",    lambda tr: mlp_predict(XYZ[tr],val[tr],qxyz)),
    ]
    print(f"WesternUS temp, Q={len(qi)} target points, median of R={a.R}  (seconds: ingest n -> predict Q)\n")
    print(f"{'method':12}{'family':7}"+"".join(f"{('n='+str(n)):>11}" for n in n_list))
    print("-"*(19+11*len(n_list)))
    for name,fam,fn in methods:
        row=[]
        for n in n_list:
            tr=rng.choice(N, size=n, replace=False)
            # skip absurdly slow cells (GP/MLP at full n) only if >60s on smaller n
            if row and row[-1]>60 and fam=="ML":
                row.append(float('nan')); continue
            row.append(timeit(lambda tr=tr: fn(tr), a.R))
        cells="".join((f"{v:>11.3f}" if v==v else f"{'skip':>11}") for v in row)
        print(f"{name:12}{fam:7}{cells}")

if __name__=="__main__": main()
