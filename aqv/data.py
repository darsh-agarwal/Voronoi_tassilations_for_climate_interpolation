"""Map projection, plus loading/cleaning for the legacy CPCB air-quality snapshot.

``project_aeqd`` is the piece every current loader uses (GHCN, GHCN-Daily, EPA
AQS); the CPCB pivot/clean helpers below serve the single-hour India snapshot
that an early version of this study used and the paper no longer reports.

The raw feed (data.gov.in / CPCB real-time) is in long format: one row per
(station, pollutant). We pivot to one row per station with a column per
pollutant, drop stations with no reading for the target pollutant, and project
geographic coordinates to a local metric plane so that Euclidean distance in
(x, y) approximates true ground distance in metres.

Projection: azimuthal equidistant about the data centroid. It preserves
distance and azimuth *from the centre point*, which is the property that matters
for distance-based interpolation over a country-sized extent. It is implemented
here in plain numpy so the repo has no geospatial dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R = 6_371_008.8  # mean Earth radius, metres (IUGG)

POLLUTANTS = ["PM2.5", "PM10", "NO2", "OZONE", "CO", "SO2", "NH3"]


def load_long(csv_path: str) -> pd.DataFrame:
    """Load the raw long-format CSV and coerce the average value to numeric."""
    df = pd.read_csv(csv_path)
    df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per physical station, one column per pollutant.

    A station is keyed by (station name, lat, lon). Duplicate rows for the same
    station/pollutant are averaged. Only the ``pollutant_avg`` value is kept.
    """
    keys = ["state", "city", "station", "latitude", "longitude"]
    wide = (
        df_long.pivot_table(
            index=keys,
            columns="pollutant_id",
            values="pollutant_avg",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def project_aeqd(lat, lon, lat0=None, lon0=None):
    """Azimuthal-equidistant projection to metres about (lat0, lon0).

    If the centre is not given it defaults to the mean of the inputs. Returns
    (x_east, y_north) arrays in metres plus the centre used (so the same centre
    can be reused for held-out points).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat0 is None:
        lat0 = float(np.mean(lat))
    if lon0 is None:
        lon0 = float(np.mean(lon))

    phi = np.radians(lat)
    lam = np.radians(lon)
    phi0 = np.radians(lat0)
    lam0 = np.radians(lon0)

    # angular distance c from the centre, and azimuth, via the standard AEQD forward eqs
    cos_c = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(lam - lam0)
    cos_c = np.clip(cos_c, -1.0, 1.0)
    c = np.arccos(cos_c)
    # k' = c / sin(c), guarding the c->0 limit (k'->1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kp = np.where(c == 0, 1.0, c / np.sin(c))
    x = EARTH_R * kp * np.cos(phi) * np.sin(lam - lam0)
    y = EARTH_R * kp * (
        np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(lam - lam0)
    )
    return x, y, (lat0, lon0)


def build_dataset(csv_path: str, pollutant: str):
    """End-to-end: return (df, xy) for one pollutant.

    ``df`` has the station rows that carry a finite reading for ``pollutant``,
    plus ``x``/``y`` metric coordinates. ``xy`` is the (N, 2) coordinate array.
    """
    wide = to_wide(load_long(csv_path))
    wide = wide[np.isfinite(wide[pollutant].values)].copy()
    x, y, centre = project_aeqd(wide["latitude"].values, wide["longitude"].values)
    wide["x"], wide["y"] = x, y
    wide.attrs["centre"] = centre
    wide.attrs["pollutant"] = pollutant
    xy = np.column_stack([x, y])
    return wide.reset_index(drop=True), xy
