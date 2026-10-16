"""Training-free spatial interpolation for sparse climate monitoring.

Importing this package loads `.aqs.env` from the repo root (if present) into the
environment, so EPA AQS credentials reach the loaders however the code is run:
directly (`python run_aq_benchmark.py`), through the shell drivers in `scripts/`,
or from a notebook. Variables already set in the environment always win.
"""
import os

_ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ".aqs.env")
if os.path.exists(_ENV):
    with open(_ENV) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
