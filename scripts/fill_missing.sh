#!/bin/zsh
set -a; . ./.aqs.env; set +a
# wait for the user's running sweep to finish (avoid file/cache contention)
while pgrep -f run_all_bakeoffs.sh >/dev/null 2>&1; do sleep 5; done
OUT=results/fill_pm25_precip.txt
: > $OUT
run() { echo "################## $1 / $2 / $3 ##################" >> $OUT
  python3 -m experiments.run_method_comparison --dataset $1 --region $2 --year $3 --max_n 400 2>&1 | grep -v -i warning >> $OUT; echo >> $OUT; }
for r in SierraNevada Colorado; do run precip $r 2023; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm25 $r 2023; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm25 $r 2021; done
echo "FILL DONE" >> $OUT
