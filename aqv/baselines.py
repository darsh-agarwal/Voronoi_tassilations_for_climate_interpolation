"""ML / DL baselines to test the thesis: can statistics beat ML/DL on sparse
spatial data? All are given the SAME information (x, y, and optionally elevation)
and tuned to a fair, sensible configuration -- no strawmen.

- Gaussian process regression == kriging (a kernel method): ordinary (x,y) and
  with elevation as an input dimension (kriging-with-drift flavour).
- Random forest and gradient boosting: standard tabular ML spatial predictors.
- MLP: a small neural net -- the lightweight deep-learning comparator. Coordinate
  MLPs have known spectral-bias limits on raw coordinates; we standardise inputs
  and give it real capacity, and note that heavier DL (Fourier features, etc.) is
  out of scope for this small-data regime.

Each function fits on (Xtr, ytr) and predicts Xq. X columns are features
(2 = x,y ; 3 = x,y,elev). Returns a plain ndarray of predictions.
"""
from __future__ import annotations
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor


def _scale(Xtr, Xq):
    s = StandardScaler().fit(Xtr)
    return s.transform(Xtr), s.transform(Xq)


def gp_predict(Xtr, ytr, Xq):
    """Gaussian process regression = kriging. RBF signal + white noise, ML-fit."""
    Xtr, Xq = _scale(Xtr, Xq)
    d = Xtr.shape[1]
    kern = C(1.0, (1e-2, 1e3)) * RBF([1.0] * d, (1e-1, 1e2)) + WhiteKernel(0.1, (1e-3, 1e1))
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=0,
                                  alpha=1e-6)
    gp.fit(Xtr, ytr)
    return gp.predict(Xq)


def rf_predict(Xtr, ytr, Xq):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=1,
                              random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xq)


def gbm_predict(Xtr, ytr, Xq):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_depth=3, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xq)


def mlp_predict(Xtr, ytr, Xq):
    Xtr, Xq = _scale(Xtr, Xq)
    base = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                        solver="lbfgs", alpha=1e-1, max_iter=2000, random_state=0)
    # standardise the target as well: a tanh net with L2=0.1 cannot fit raw
    # precip/PM10 magnitudes (up to ~4600 mm) without the penalty crushing the
    # output weights -- unfair to the DL baseline. Scale y, invert on predict.
    m = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    m.fit(Xtr, ytr)
    return m.predict(Xq)
