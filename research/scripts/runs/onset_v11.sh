cd ~/Documents/chimera-vocal-synth
R=~/Documents/chimera-vocal-synth-results/onset_v11
for off in -10 -5 5 10; do
 PROBE_WORDS=kita,kata,kɯta,gida,gada,gɯda PROBE_ONSET_ONLY=1 PROBE_DURS=150 PROBE_GLOTTIS=folds,cutoff PROBE_YIN_MIN=60 PROBE_FOLDS_LOOP=frozen PROBE_LOCK_RELEASE=1 PROBE_LOCK_OFFSET_MS=$off $R/probe.bin > $R/off$off.tsv 2>> $R/err
done
nice -n 10 cargo test --release -q --workspace > $R/tests.log 2>&1; echo "tests exit $?" >> $R/tests.log
echo done > $R/status
