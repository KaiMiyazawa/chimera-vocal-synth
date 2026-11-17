#!/bin/sh
# CHIMERA の測定を上限つきで起動する（2026-11-16 本人の指示: WSL を止めないことを速さより優先）。
# 使い方: run-capped.sh UNIT_NAME 'コマンド'   例: run-capped.sh chimera-v10 "cd DIR && ./probe.bin > out.tsv"
# 上限: CPU 4 コア分（16 のうち）、nice 10、メモリ 3GB。同時に走らせる測定は全部で 1 ユニットにまとめる。
set -eu
unit=$1; shift
exec systemd-run --user --unit="$unit" --collect \
  -p CPUQuota=400% -p Nice=10 -p MemoryMax=3G -p MemorySwapMax=0 \
  -p WorkingDirectory="$HOME/Documents/chimera-vocal-synth" \
  sh -c "$*"
