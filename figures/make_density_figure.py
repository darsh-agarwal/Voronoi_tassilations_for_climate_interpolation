"""Figure: how sparse are these networks, and how are the stations spread?

Panels (a)-(c): three contrasting temperature networks in the projected metric
space the interpolators actually work in, with the Voronoi tessellation each
station induces (the natural-neighbour catchment). Panel (d): median
nearest-station distance against network size for all 27 region-year benchmarks.

Networks are loaded exactly as run_method_comparison.py loads them (same
max_n=400 subsample, same seed), so n matches the appendix table.

    python make_density_figure.py        -> fig_density.{pdf,png}
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # TrueType, not Type 3 (venue requirement)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree

from aqv.data import project_aeqd
from experiments.run_method_comparison import load_data, BOX, EPA_REGIONS

MAPS = [("temp", "Colorado", 2024), ("temp", "Alps", 2024), ("temp", "Himalaya", 2024)]
ALL = ([("temp", r, 2024) for r in BOX]
       + [("precip", r, 2023) for r in ("PacNW", "SierraNevada", "Colorado")]
       + [(d, r, y) for d in ("pm25", "pm10") for r in EPA_REGIONS for y in (2023, 2021)])
STYLE = {"temp": ("#c1442c", "o", "Temperature"), "precip": ("#2b6fa8", "s", "Precipitation"),
         "pm25": ("#3f8f54", "^", "PM$_{2.5}$"), "pm10": ("#8a6bb1", "v", "PM$_{10}$")}


def network(dataset, region, year, max_n=400):
    """(x_km, y_km, elev, n, median nearest-station distance km) as benchmarked."""
    lat, lon, z, val, _, _ = load_data(dataset, region, year)
    if len(val) > max_n:                      # same subsample as the bake-off
        sub = np.random.default_rng(0).choice(len(val), max_n, replace=False)
        lat, lon, z = lat[sub], lon[sub], z[sub]
    x, y, _ = project_aeqd(lat, lon)
    xy = np.column_stack([x, y])
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1] / 1000.0
    return x / 1000.0, y / 1000.0, z, len(z), float(np.median(nn))


def _outdir():
    """paper/ when working in the full tree, figures/out/ for a plain clone."""
    import os
    d = "paper" if os.path.isdir("paper") else os.path.join("figures", "out")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    nets = {k: network(*k) for k in MAPS}
    vmin, vmax = 0, max(np.max(v[2]) for v in nets.values())

    fig = plt.figure(figsize=(11.0, 2.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.30,
                          left=0.045, right=0.675, bottom=0.19, top=0.87)
    gsd = fig.add_gridspec(1, 1, left=0.790, right=0.985, bottom=0.19, top=0.87)

    axes = []
    for i, key in enumerate(MAPS):
        ax = fig.add_subplot(gs[0, i])
        xk, yk, z, n, nn = nets[key]
        pts = np.column_stack([xk, yk])
        if 4 <= n < 100:  # cell edges only on sparse panels; on n=348 they smear the colours
            voronoi_plot_2d(Voronoi(pts), ax=ax, show_points=False, show_vertices=False,
                            line_colors="#b9bfc6", line_width=0.45, line_alpha=0.9)
        sc = ax.scatter(xk, yk, c=z, s=15 if n < 100 else 7, cmap="terrain",
                        vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.25, zorder=3)
        pad = 0.06 * max(np.ptp(xk), np.ptp(yk))
        ax.set_xlim(xk.min() - pad, xk.max() + pad)
        ax.set_ylim(yk.min() - pad, yk.max() + pad)
        # equal aspect without ragged panel sizes: stretch the data limits, not the box
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"({'abc'[i]}) {key[1]}  n={n}", fontsize=9, pad=3)
        ax.text(0.03, 0.04, f"median spacing {nn:.0f} km", transform=ax.transAxes,
                fontsize=7.2, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.4))
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("km", fontsize=7, labelpad=1)
        if i == 0:
            ax.set_ylabel("km", fontsize=7, labelpad=1)
        axes.append(ax)

    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("station elevation (m)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gsd[0, 0])
    seen = set()
    for ds, reg, yr in ALL:
        try:
            _, _, _, n, nn = network(ds, reg, yr)
        except Exception as e:
            print(f"  skip {ds}/{reg}/{yr}: {type(e).__name__}")
            continue
        c, m, lab = STYLE[ds]
        ax.scatter(n, nn, c=c, marker=m, s=26, alpha=0.85, edgecolors="none",
                   label=None if ds in seen else lab)
        seen.add(ds)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("stations in network", fontsize=8)
    ax.set_ylabel("median spacing (km)", fontsize=8, labelpad=2)
    ax.set_title("(d) all 27 benchmarks", fontsize=9, pad=3)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=6.6, frameon=False, loc="upper right", handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)

    out = _outdir()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_density.{ext}"), dpi=240)
    print(f"wrote {out}/fig_density.pdf and .png")


if __name__ == "__main__":
    main()
