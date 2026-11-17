#!/bin/bash -l
R=~/Documents/chimera-vocal-synth-results/compare
for w in upstream-main fork-main; do
  cd /hdd/chimera-wt/$w || exit 1
  for k in tsp csd; do
    out=$R/${w}_$k
    if [ $k = tsp ]; then ds=~/datasets/tsp; else ds=~/datasets/csd/CSD; fi
    ./target/release/chimera compare $k $ds $out --jobs 4 > $out.log 2>&1
    echo "$w $k exit $?" >> $R/status
  done
done
echo "exit done" >> $R/status
