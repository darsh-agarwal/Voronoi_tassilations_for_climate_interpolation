#!/bin/zsh
set -a; . ./.aqs.env; set +a
OUT=results/bakeoff_all.txt
# : > $OUT
run() {  # dataset region year
  echo "################## $1 / $2 / $3 ##################" >> $OUT
  python3 -m experiments.run_method_comparison --dataset $1 --region $2 --year $3 --max_n 400 2>&1 \
    | grep -v -i warning >> $OUT
  echo >> $OUT
}
# for r in Colorado WesternUS Alps Himalaya Andes Caucasus SierraNevada PacNW; do run temp $r 2024; done
# for r in PacNW SierraNevada Colorado; do run precip $r 2023; done
# for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm25 $r 2023; done
# for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm25 $r 2021; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm10 $r 2023; done
for r in SouthernRockies NorthernRockies SierraGreatBasin Cascades; do run pm10 $r 2021; done
echo "PM10 ADDITIONS DONE" >> $OUT
