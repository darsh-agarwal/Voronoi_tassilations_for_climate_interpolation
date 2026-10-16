"""Leave-one-station-out cross-validation, error metrics, and a paired test.

LOOCV is the standard fair test for spatial interpolation: hold out one station,
predict its value from all the others, repeat for every station, then compare
predictions to the withheld truth. Because IDW and NNI are both memory-based
(no fitted parameters beyond the neighbourhood settings), LOOCV here is exact,
not an approximation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def metrics(true, pred, mape_floor=5.0):
    """Standard interpolation error metrics.

    RMSE, MAE and mean bias are in the pollutant's own units. MAPE is computed
    only over points whose true value exceeds ``mape_floor`` (percentage error
    explodes near zero and would otherwise be meaningless for trace readings).
    R2 is the coefficient of determination against the mean predictor.
    """
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    m = true > mape_floor
    mape = float(np.mean(np.abs(err[m] / true[m])) * 100) if m.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2, "mape": mape,
            "n": int(len(true))}


def loocv(xy, values, predict_fn):
    """Leave-one-station-out predictions for every station.

    ``predict_fn(train_xy, train_val, query_xy)`` may return either an array of
    predictions, or a (predictions, interior_mask) tuple (as NNI does). Returns
    (pred, interior) where ``interior`` marks points the method handled natively
    (True everywhere for methods that don't distinguish).
    """
    xy = np.asarray(xy, float)
    values = np.asarray(values, float)
    n = len(values)
    pred = np.empty(n)
    interior = np.ones(n, dtype=bool)
    all_idx = np.arange(n)
    for i in range(n):
        mask = all_idx != i
        res = predict_fn(xy[mask], values[mask], xy[i:i + 1])
        if isinstance(res, tuple):
            p, inside = res
            pred[i] = p[0]
            interior[i] = bool(inside[0])
        else:
            pred[i] = res[0]
    return pred, interior


def paired_test(true, pred_a, pred_b):
    """Wilcoxon signed-rank test on per-station absolute errors (A vs B).

    Tests whether method A's absolute errors are systematically smaller than
    B's across stations -- the honest "is the improvement real, or noise?"
    check. Returns (median_abs_err_a, median_abs_err_b, statistic, p_value).
    """
    ea = np.abs(np.asarray(pred_a, float) - np.asarray(true, float))
    eb = np.abs(np.asarray(pred_b, float) - np.asarray(true, float))
    diff = ea - eb
    nz = diff != 0
    if nz.sum() < 5:
        return float(np.median(ea)), float(np.median(eb)), float("nan"), float("nan")
    stat, p = stats.wilcoxon(ea[nz], eb[nz])
    return float(np.median(ea)), float(np.median(eb)), float(stat), float(p)


def nearest_neighbor_dist(xy):
    """Distance from each station to its nearest OTHER station (metres).

    In LOOCV this is exactly the distance from each held-out point to the
    closest training station -- i.e. how 'under-monitored' that location is.
    The whole value proposition of a better interpolator is that it should
    help MORE where this distance is large (sparse/remote), so we stratify
    error by it.
    """
    xy = np.asarray(xy, float)
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)      # k=1 is self (d=0)
    return d[:, 1]


def stratify_by_distance(true, preds, xy, qs=(0.5, 0.75, 0.9)):
    """RMSE per method within distance-to-nearest-station strata.

    ``preds`` is {method_name: prediction_array}. Splits stations by quantiles
    of nearest-neighbour distance and reports each method's RMSE in each
    stratum, so you can show the gain widening as monitoring thins out.
    Returns a list of dicts, one per stratum (near -> far).
    """
    true = np.asarray(true, float)
    d = nearest_neighbor_dist(xy)
    edges = [d.min() - 1] + [float(np.quantile(d, q)) for q in qs] + [d.max() + 1]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d > lo) & (d <= hi)
        if m.sum() == 0:
            continue
        row = {"dist_lo_km": round(max(lo, 0) / 1000, 1),
               "dist_hi_km": round(hi / 1000, 1),
               "n": int(m.sum())}
        for name, p in preds.items():
            e = np.asarray(p, float)[m] - true[m]
            row[name] = float(np.sqrt(np.mean(e**2)))
        rows.append(row)
    return rows
