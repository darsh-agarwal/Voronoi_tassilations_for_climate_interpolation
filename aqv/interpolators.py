"""Interpolators: the IDW baseline and discrete-Sibson natural-neighbour.

Both expose the same call shape:

    predict(train_xy, train_val, query_xy) -> np.ndarray

so the cross-validation harness can treat them interchangeably.

Inverse-distance weighting (IDW) is the baseline from the reference paper:
neighbours are found with a KD-tree, weighted by 1 / d**power.

Natural-neighbour interpolation is implemented via **discrete Sibson**
(Park, Linsen, Kreylos, Owens, Hamann 2006): instead of building an explicit
Voronoi diagram and clipping polygons, we rasterise a small grid around the
query point and count, cell by cell, how much "territory" the query point
steals from each existing site when it is inserted. The fraction stolen from
site i is its Sibson weight. This realises exactly the Voronoi area/volume-
stealing definition of natural-neighbour interpolation, is robust (no geometric
degeneracies to special-case), and generalises unchanged to 3-D by swapping the
2-D pixel grid for a 3-D voxel grid.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Inverse distance weighting (baseline)
# --------------------------------------------------------------------------- #
def idw_predict(train_xy, train_val, query_xy, power=2.0, k=8, radius=None,
                eps=1e-9):
    """Predict by inverse-distance weighting.

    Neighbours: the ``k`` nearest by default, or every site within ``radius``
    metres if ``radius`` is given (the KD-tree search-radius form used in the
    reference paper). Weight of neighbour i is 1 / d_i**power. An exact hit
    (d < eps) returns that site's value directly.
    """
    train_xy = np.asarray(train_xy, float)
    train_val = np.asarray(train_val, float)
    query_xy = np.asarray(query_xy, float)
    tree = cKDTree(train_xy)
    out = np.empty(len(query_xy))

    if radius is not None:
        idx_lists = tree.query_ball_point(query_xy, r=radius)
        for i, q in enumerate(query_xy):
            idx = idx_lists[i]
            if not idx:  # nothing in radius -> fall back to nearest site
                _, j = tree.query(q, k=1)
                out[i] = train_val[j]
                continue
            d = np.linalg.norm(train_xy[idx] - q, axis=1)
            if d.min() < eps:
                out[i] = train_val[idx][d.argmin()]
                continue
            w = 1.0 / d**power
            out[i] = np.dot(w, train_val[idx]) / w.sum()
        return out

    kk = min(k, len(train_xy))
    d, idx = tree.query(query_xy, k=kk)
    d = np.atleast_2d(d)
    idx = np.atleast_2d(idx)
    for i in range(len(query_xy)):
        di, ii = d[i], idx[i]
        if di.min() < eps:
            out[i] = train_val[ii][di.argmin()]
            continue
        w = 1.0 / di**power
        out[i] = np.dot(w, train_val[ii]) / w.sum()
    return out


# --------------------------------------------------------------------------- #
# Natural-neighbour interpolation via discrete Sibson
# --------------------------------------------------------------------------- #
def _sibson_weights_2d(q, sites_xy, grid_res=64, span_k=8, span_scale=1.6):
    """Discrete Sibson weights for a single query point q in 2-D.

    ``sites_xy`` are the *candidate* natural neighbours (a local subset of the
    data is enough — true natural neighbours are always nearby). We build a
    square grid around q, assign each grid cell to its nearest site both with
    and without q present, and credit each site the cells q takes from it.

    Returns (weights, valid). ``valid`` is False when q steals no territory
    (grid too coarse / q coincident with a site) or when the stolen region hits
    the grid boundary (q is effectively outside the sites' convex hull, so the
    weights are unreliable) -> the caller should fall back.
    """
    sites_xy = np.asarray(sites_xy, float)
    d_to_sites = np.linalg.norm(sites_xy - q, axis=1)
    # grid half-width scaled by the distance to the span_k-th nearest site
    ref = np.sort(d_to_sites)[min(span_k, len(d_to_sites)) - 1]
    if ref <= 0:
        return None, False
    h = span_scale * ref
    lin = np.linspace(-h, h, grid_res)
    gx, gy = np.meshgrid(lin, lin)
    cells = np.column_stack([gx.ravel() + q[0], gy.ravel() + q[1]])

    # nearest site WITHOUT q
    tree = cKDTree(sites_xy)
    d_wo, owner_wo = tree.query(cells, k=1)
    # distance to q
    d_q = np.linalg.norm(cells - q, axis=1)
    stolen = d_q < d_wo  # cells that switch to q when q is inserted
    if not stolen.any():
        return None, False

    # boundary check: if q's territory touches the grid edge it is unbounded
    edge = np.zeros(grid_res * grid_res, dtype=bool)
    edge_mask = np.zeros((grid_res, grid_res), dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    edge = edge_mask.ravel()
    if stolen[edge].any():
        return None, False

    owners = owner_wo[stolen]
    w = np.bincount(owners, minlength=len(sites_xy)).astype(float)
    total = w.sum()
    if total == 0:
        return None, False
    return w / total, True


def _sibson_weights_3d(q, sites_xyz, grid_res=28, span_k=8, span_scale=1.6):
    """Discrete Sibson weights for a query point q in 3-D (anisotropic space).

    Coordinates are the anisotropy-embedded (x, y, beta*z). Air-quality
    "stations" lie on the terrain surface -- a 2-manifold in 3-D -- so a query
    point's stolen Voronoi volume is naturally unbounded perpendicular to the
    surface. We therefore clip the voxel grid's vertical extent to the local
    span of candidate elevations and treat only the horizontal (x, y) faces as
    hull boundaries. This keeps the stolen volume finite and meaningful (the
    "surface slab") instead of degenerating on near-coplanar points.
    """
    sites = np.asarray(sites_xyz, float)
    d = np.linalg.norm(sites - q, axis=1)
    ref = np.sort(d)[min(span_k, len(d)) - 1]
    if ref <= 0:
        return None, False
    hxy = span_scale * ref
    # vertical extent: local span of candidate z (embedded), with a small margin
    zmin, zmax = sites[:, 2].min(), sites[:, 2].max()
    zpad = 0.15 * (zmax - zmin) + 1e-6
    zlo, zhi = min(zmin, q[2]) - zpad, max(zmax, q[2]) + zpad

    lin = np.linspace(-hxy, hxy, grid_res)
    zlin = np.linspace(zlo, zhi, grid_res)
    gx, gy, gz = np.meshgrid(lin + q[0], lin + q[1], zlin, indexing="ij")
    cells = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    tree = cKDTree(sites)
    d_wo, owner_wo = tree.query(cells, k=1)
    d_q = np.linalg.norm(cells - q, axis=1)
    stolen = d_q < d_wo
    if not stolen.any():
        return None, False

    # horizontal-face boundary check only (z faces are the surface slab, allowed)
    g = grid_res
    face = np.zeros((g, g, g), dtype=bool)
    face[0, :, :] = face[-1, :, :] = face[:, 0, :] = face[:, -1, :] = True
    if stolen[face.ravel()].any():
        return None, False

    w = np.bincount(owner_wo[stolen], minlength=len(sites)).astype(float)
    total = w.sum()
    if total == 0:
        return None, False
    return w / total, True


def nni_predict_resid(train_xy, train_val, train_z, query_xy, query_z,
                      degree=1, **nni_kw):
    """Elevation as a covariate: fit value ~ poly(elevation), interpolate the
    residual with 2-D NNI, add the trend back (regression / residual NNI).

    Degrades gracefully -- if elevation carries no signal the fitted trend is
    ~flat and this reduces to plain 2-D NNI, so it cannot hurt much. Robust to
    the anisotropy mis-estimation that can trip the 3-D embedding.
    """
    train_z = np.asarray(train_z, float); query_z = np.asarray(query_z, float)
    train_val = np.asarray(train_val, float)
    coef = np.polyfit(train_z, train_val, degree)
    trend_tr = np.polyval(coef, train_z)
    trend_q = np.polyval(coef, query_z)
    resid = train_val - trend_tr
    rpred, interior = nni_predict(train_xy, resid, query_xy, **nni_kw)
    return trend_q + rpred, interior


def nni_predict_3d(train_xy, train_val, train_z, query_xy, query_z, beta,
                   n_candidates=24, grid_res=28, span_k=8, span_scale=1.6,
                   fallback=None):
    """Elevation-aware NNI: discrete Sibson in anisotropic (x, y, beta*z) space.

    ``beta`` is the anisotropy factor (typically from ``variogram``). Falls back
    (default: 2-D NNI ignoring elevation) for query points whose 3-D Sibson
    region is horizontally unbounded or unresolved.
    """
    train_xy = np.asarray(train_xy, float); query_xy = np.asarray(query_xy, float)
    tz = np.asarray(train_z, float) * beta
    qz = np.asarray(query_z, float) * beta
    train_val = np.asarray(train_val, float)
    sites = np.column_stack([train_xy, tz])
    queries = np.column_stack([query_xy, qz])

    tree = cKDTree(sites)
    m = min(n_candidates, len(sites))
    _, cand_idx = tree.query(queries, k=m)
    cand_idx = np.atleast_2d(cand_idx)

    out = np.empty(len(queries))
    interior = np.zeros(len(queries), dtype=bool)
    fb_idx = []
    for i in range(len(queries)):
        idx = cand_idx[i]
        w, ok = _sibson_weights_3d(queries[i], sites[idx], grid_res, span_k, span_scale)
        if ok:
            out[i] = np.dot(w, train_val[idx])
            interior[i] = True
        else:
            fb_idx.append(i)

    if fb_idx:
        if fallback is None:
            fb, _ = nni_predict(train_xy, train_val, query_xy[fb_idx])
        else:
            fb = fallback(train_xy, train_val, query_xy[fb_idx])
        for j, i in enumerate(fb_idx):
            out[i] = fb[j]
    return out, interior


def nni_predict(train_xy, train_val, query_xy, n_candidates=24,
                grid_res=64, span_k=8, span_scale=1.6, fallback=idw_predict):
    """Predict by discrete-Sibson natural-neighbour interpolation.

    For each query point we take its ``n_candidates`` nearest sites as the
    local support, compute Sibson weights on a rasterised grid, and return the
    weighted average of their values. Query points whose Sibson region is
    unbounded (outside the convex hull) or unresolved fall back to ``fallback``
    (IDW by default), and the boolean mask of hull-interior points is returned
    so the harness can report them separately.
    """
    train_xy = np.asarray(train_xy, float)
    train_val = np.asarray(train_val, float)
    query_xy = np.asarray(query_xy, float)
    tree = cKDTree(train_xy)
    m = min(n_candidates, len(train_xy))
    _, cand_idx = tree.query(query_xy, k=m)
    cand_idx = np.atleast_2d(cand_idx)

    out = np.empty(len(query_xy))
    interior = np.zeros(len(query_xy), dtype=bool)
    fallback_idx = []
    for i, q in enumerate(query_xy):
        idx = cand_idx[i]
        w, ok = _sibson_weights_2d(q, train_xy[idx], grid_res, span_k, span_scale)
        if ok:
            out[i] = np.dot(w, train_val[idx])
            interior[i] = True
        else:
            fallback_idx.append(i)

    if fallback_idx:
        fb = fallback(train_xy, train_val, query_xy[fallback_idx])
        for j, i in enumerate(fallback_idx):
            out[i] = fb[j]
    return out, interior
