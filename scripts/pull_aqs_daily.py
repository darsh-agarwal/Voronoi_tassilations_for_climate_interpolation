"""Pull EPA AQS *daily* PM2.5 data for all states needed for the ST-GNN paper.

States covered:
  Colorado(08), Utah(49), Wyoming(56), New Mexico(35), Arizona(04)  -> Southern/Northern Rockies
  Idaho(16), Montana(30)                                             -> Northern Rockies
  California(06), Nevada(32)                                         -> Sierra/Great Basin
  Washington(53), Oregon(41)                                         -> Cascades

Years: 2023 (normal) and 2021 (wildfire). Both in same run.

Output: data/epa_aqs_daily/daily_{state}_{param}_{year}.json
Runtime: ~10-20 min depending on API rate limits.

Usage:
    # credentials live in voronoi/.aqs.env (AQS_EMAIL=... / AQS_KEY=...),
    # loaded automatically below; never commit that file or hardcode the key.
    python -m scripts.pull_aqs_daily
"""
# (aqv/__init__.py loads .aqs.env on import)

from aqv.loaders.aqs_daily import pull_aqs_daily

STATES = ["08", "49", "56", "35", "04", "16", "30", "06", "32", "53", "41"]
YEARS  = [2023, 2021]
PARAMS = ("88101",)      # FRM/FEM only; 88502 adds noise, skip for speed

if __name__ == "__main__":
    for year in YEARS:
        print(f"\n{'='*50}")
        print(f"  Pulling AQS daily PM2.5 for year={year}")
        print(f"{'='*50}")
        pull_aqs_daily(year=year, states=STATES, params=PARAMS)
    print("\nDone. Files in data/epa_aqs_daily/")
