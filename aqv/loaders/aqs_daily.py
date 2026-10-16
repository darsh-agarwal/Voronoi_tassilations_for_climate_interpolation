"""EPA AQS *daily* PM2.5 download and cache.

Separate from the annual loader (`aqv.loaders.epa_aqs`) because the daily series
is only used by the episode-scale wildfire analysis, and the files are large
(hundreds of MB per state-year). Credentials come from AQS_EMAIL / AQS_KEY,
which `aqv/__init__.py` loads from `.aqs.env`.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

AQS_DAILY_DIR = os.path.join("data", "epa_aqs_daily")
AQS_BASE = "https://aqs.epa.gov/data/api/dailyData/byState"


def _fetch_aqs_daily(email: str, key: str, state: str, year: int,
                     param: str = "88101",
                     retries: int = 5, pause: float = 4.0) -> list:
    """Pull dailyData/byState for one state/year, cached to AQS_DAILY_DIR."""
    cp = os.path.join(AQS_DAILY_DIR, f"daily_{state}_{param}_{year}.json")
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f).get("Data", [])

    q = urllib.parse.urlencode({
        "email": email, "key": key, "param": param, "state": state,
        "bdate": f"{year}0101", "edate": f"{year}1231",
    })
    url = f"{AQS_BASE}?{q}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                payload = json.load(r)
            os.makedirs(AQS_DAILY_DIR, exist_ok=True)
            with open(cp, "w") as f:
                json.dump(payload, f)
            print(f"  fetched state={state} param={param} year={year} "
                  f"({len(payload.get('Data', []))} rows)")
            return payload.get("Data", [])
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED state={state}: {e}")
                return []
            wait = pause * (2 ** attempt)
            print(f"  retry {attempt+1} for state={state} in {wait:.0f}s: {e}")
            time.sleep(wait)
    return []


def pull_aqs_daily(year: int, states: list[str],
                   params: tuple[str, ...] = ("88101",),
                   email: str | None = None, key: str | None = None) -> None:
    """Download and cache EPA AQS daily PM2.5 for all state/year combos.

    Call this once; subsequent calls use the cache. Rate-limited to ~1 req/4s.
    """
    email = email or os.environ.get("AQS_EMAIL")
    key = key or os.environ.get("AQS_KEY")
    if not email or not key:
        raise RuntimeError("Set AQS_EMAIL and AQS_KEY env vars.")
    total = len(states) * len(params)
    done = 0
    for param in params:
        for state in states:
            done += 1
            print(f"[{done}/{total}] state={state} param={param} year={year}")
            _fetch_aqs_daily(email, key, state, year, param)
            time.sleep(4.0)      # AQS rate limit: ~1 req/sec; be conservative
