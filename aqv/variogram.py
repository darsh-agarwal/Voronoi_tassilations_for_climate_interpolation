"""Directional empirical variogram and anisotropy-factor estimation.

The vertical dimension (elevation) decorrelates far faster than the horizontal
plane: pollution changes over tens-to-hundreds of metres of altitude but over
kilometres horizontally. Before embedding elevation as a third coordinate we
must rescale it, or Euclidean distance in raw (x, y, z) treats a 100 m climb as
"as close" as 100 m along the ground, which is physically wrong.

We estimate the anisotropy factor beta = (horizontal range) / (vertical range)
empirically: the ratio of separation distances at which the semivariance reaches
a fixed fraction of its sill. Points are then embedded as (x, y, beta * z).
"""
from __future__ import annotations

import numpy as np


def _semivariance_by_lag(sep, dval2, n_bins, max_lag):
    """Mean semivariance 0.5*E[(v_i-v_j)^2] in bins of separation ``sep``."""
    edges = np.linspace(0, max_lag, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (sep >= edges[b]) & (sep < edges[b + 1])
        if m.sum() >= 30:
            gamma[b] = 0.5 * np.mean(dval2[m])
    return centres, gamma


def _range_at(centres, gamma, sill, frac=0.95):
    """First lag at which gamma reaches frac*sill (linear interp)."""
    ok = np.isfinite(gamma)
    c, g = centres[ok], gamma[ok]
    target = frac * sill
    above = np.where(g >= target)[0]
    if len(above) == 0:
        return float(c[-1]) if len(c) else np.nan
    i = above[0]
    if i == 0:
        return float(c[0])
    # interpolate between i-1 and i
    g0, g1, c0, c1 = g[i - 1], g[i], c[i - 1], c[i]
    if g1 == g0:
        return float(c1)
    return float(c0 + (target - g0) * (c1 - c0) / (g1 - g0))


def anisotropy_factor(xy, z, val, n_bins=15, sample_pairs=200_000,
                      rng=None, default=25.0):
    """Estimate beta = horizontal_range / vertical_range from the data.

    ``xy`` metric horizontal coords, ``z`` elevation (m), ``val`` values.
    Pairs are subsampled for speed. Returns (beta, info) where ``info`` holds
    the fitted ranges and curves. Falls back to ``default`` if the vertical
    spread is too small to fit (near-flat region).
    """
    rng = rng or np.random.default_rng(0)
    xy = np.asarray(xy, float); z = np.asarray(z, float); val = np.asarray(val, float)
    n = len(val)
    # sample random pairs
    m = min(sample_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    ok = i != j
    i, j = i[ok], j[ok]
    dh = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
    dz = np.abs(z[i] - z[j])
    dv2 = (val[i] - val[j]) ** 2
    sill = float(np.var(val))

    ch, gh = _semivariance_by_lag(dh, dv2, n_bins, np.percentile(dh, 90))
    a_h = _range_at(ch, gh, sill)

    # Vertical range must be estimated from pairs that are CLOSE horizontally,
    # otherwise horizontal separation leaks into the vertical variogram and
    # fakes a tiny vertical range (huge beta). We keep only pairs within a small
    # horizontal band and look at how semivariance grows with elevation gap.
    h_band = np.percentile(dh, 15)
    close = dh < h_band
    if np.ptp(z) < 50 or close.sum() < 500 or np.percentile(dz[close], 90) < 30:
        return default, {"a_h": a_h, "a_v": None, "reason": "flat-or-few-pairs"}
    cv, gv = _semivariance_by_lag(dz[close], dv2[close], n_bins,
                                  np.percentile(dz[close], 90))
    a_v = _range_at(cv, gv, sill)

    if not (np.isfinite(a_h) and np.isfinite(a_v)) or a_v <= 0:
        return default, {"a_h": a_h, "a_v": a_v, "reason": "fit-failed"}
    beta = float(np.clip(a_h / a_v, 1.0, 100.0))
    return beta, {"a_h": a_h, "a_v": a_v, "beta_raw": a_h / a_v,
                  "h_curve": (ch, gh), "v_curve": (cv, gv), "sill": sill,
                  "h_band": h_band}
