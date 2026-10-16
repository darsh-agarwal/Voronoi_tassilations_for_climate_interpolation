"""Figure: horizontal vs vertical variogram for Colorado temperature.

Regenerates fig_variogram.{pdf,png} from the same estimator the 3-D
embedding uses (aqv.variogram.anisotropy_factor), so the plotted ranges are
exactly the a_h and a_v that set beta.

    python make_variogram_figure.py
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

from aqv.loaders.ghcn import region_dataset
from aqv.data import project_aeqd
from aqv.variogram import anisotropy_factor

BOX_COLORADO = (36, 41, -109, -102)


def _outdir():
    """paper/ when working in the full tree, figures/out/ for a plain clone."""
    import os
    d = "paper" if os.path.isdir("paper") else os.path.join("figures", "out")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    d = region_dataset(2024, BOX_COLORADO)
    x, y, _ = project_aeqd(d["lat"], d["lon"])
    xy = np.column_stack([x, y])
    beta, info = anisotropy_factor(xy, d["elev"], d["temp"])
    a_h, a_v, sill = info["a_h"], info["a_v"], info["sill"]
    (ch, gh), (cv, gv) = info["h_curve"], info["v_curve"]
    print(f"n={d['n']}  a_h={a_h/1000:.0f} km  a_v={a_v:.0f} m  "
          f"beta_raw={info['beta_raw']:.0f}  beta={beta:.0f}  sill={sill:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for ax, (c, g, rng, unit, scale, title) in zip(axes, [
            (ch, gh, a_h, "km", 1e-3, "horizontal separation"),
            (cv, gv, a_v, "m", 1.0, "vertical separation (elevation)")]):
        ok = np.isfinite(g)
        ax.plot(c[ok] * scale, g[ok], "o-", ms=3.5, lw=1.2, color="#2b6fa8")
        ax.axhline(sill, ls=":", lw=1.0, color="#777",
                   label=f"sill ({sill:.1f})")
        ax.axvline(rng * scale, ls="--", lw=1.0, color="#c1442c",
                   label=f"range $\\approx$ {rng*scale:.0f} {unit}")
        ax.set_xlabel(f"{title} ({unit})", fontsize=8.5)
        ax.set_ylabel("semivariance $\\gamma(h)$", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[0].set_title("(a) horizontal", fontsize=9)
    axes[1].set_title("(b) vertical", fontsize=9)
    fig.suptitle(f"Colorado temperature, {d['n']} GHCN stations "
                 f"($\\beta = a_h/a_v$ clipped to {beta:.0f})", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = _outdir()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_variogram.{ext}"), dpi=240)
    print(f"wrote {out}/fig_variogram.pdf and .png")


if __name__ == "__main__":
    main()
