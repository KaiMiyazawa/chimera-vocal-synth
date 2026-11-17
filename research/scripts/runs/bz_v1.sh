cd ~/Documents/chimera-vocal-synth
R=~/Documents/chimera-vocal-synth-results/bz_v1
$R/probe.bin --bz > $R/linear.tsv 2>> $R/err
$R/probe.bin --bz --cubic > $R/cubic.tsv 2>> $R/err
cargo test --release -q --workspace > $R/tests.log 2>&1; echo "tests exit $?" >> $R/tests.log
echo done > $R/status
