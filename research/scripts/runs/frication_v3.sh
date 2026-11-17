cd ~/Documents/chimera-vocal-synth
R=~/Documents/chimera-vocal-synth-results/frication_v3
for j in 0 1; do PROBE_JUNCTION=$j PROBE_AG=0 PROBE_TB_DY=0,-0.25,-0.5,-1 PROBE_DURS=100,140 $R/probe.bin > $R/j$j.tsv 2>> $R/err; done
echo done > $R/status
