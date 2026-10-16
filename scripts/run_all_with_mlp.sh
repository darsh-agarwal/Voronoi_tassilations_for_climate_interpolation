#!/bin/zsh
# Re-run all 27 bake-offs WITH the MLP enabled, dumping a fresh machine-readable CSV.
set -a; . ./.aqs.env 2>/dev/null; set +a
export BAKEOFF_CSV=results/tables/all_methods_new.csv
rm -f "$BAKEOFF_CSV"
LOG=results/bakeoff_mlp.log
: > "$LOG"
run(){ echo "## $1 $2 $3" | tee -a "$LOG"; python3 -m experiments.run_method_comparison --dataset $1 --region $2 --year $3 --max_n 400 >>"$LOG" 2>&1; }
for r in Colorado WesternUS Alps Himalaya Andes Caucasus SierraNevada PacNW; do run temp $r 2024; done
for r in PacNW SierraNevada Colorado; do run precip $r 2023; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm25 $r 2023; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm25 $r 2021; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm10 $r 2023; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm10 $r 2021; done
echo "ALL DONE rows=$(($(wc -l < "$BAKEOFF_CSV")-1))" | tee -a "$LOG"
