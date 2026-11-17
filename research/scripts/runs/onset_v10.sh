cd ~/Documents/chimera-vocal-synth
R=~/Documents/chimera-vocal-synth-results/onset_v10
for lock in 0 1; do for ra in 0.005 0.01 0.02; do
 PROBE_WORDS=kita,kata,kɯta,gida,gada,gɯda PROBE_ONSET_ONLY=1 PROBE_DURS=150 PROBE_GLOTTIS=folds,cutoff PROBE_YIN_MIN=60 PROBE_FOLDS_LOOP=frozen PROBE_LOCK_RELEASE=$lock PROBE_RELEASE_AREA=$ra $R/probe.bin > $R/lock${lock}_ra${ra}.tsv 2>> $R/err
done; done
echo done > $R/status
