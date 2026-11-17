#!/bin/bash -l
set -u
R=~/Documents/chimera-vocal-synth-results
for w in upstream-main fork-main; do
  cd /hdd/chimera-wt/$w || exit 1
  out=$R/wer/${w/-/_}
  git log --oneline -1 > $out/commit.txt
  cargo build -q --release -p chimera-cli --jobs 4 > $out/build.log 2>&1 || { echo "build fail $w" >> $R/wer/baseline.status; continue; }
  ~/.cache/chimera-asr/.venv/bin/python tools/asr/speech_wer.py $out > $out/wer.log 2>&1
  echo "$w exit $?" >> $R/wer/baseline.status
done
echo exit done >> $R/wer/baseline.status
