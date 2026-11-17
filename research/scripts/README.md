# 研究用の測定と集計のスクリプト

このフォークの研究（日本語の母音の無声化・子音の後の F0 の揺れ）で使った、測定の起動と集計のスクリプト。エンジンの測定プログラム本体は `crates/chimera/examples/*_probe.rs` にある。

**結果のデータ・音声はここに置かない。** 測定の出力（TSV）は手元の `~/Documents/chimera-vocal-synth-results/<名前>/` に書き出す。JVS・CSD・TSP の音声と、そこから作った表（例: `jvs_cf0.py` の `tokens.tsv`）は再配布できないので、push しない。

## 起動

- `run-capped.sh UNIT 'コマンド'`: systemd の user ユニットとして、上限つきで起動する（CPU 4 コア分・nice 10・メモリ 3 GB・swap なし）。同時に走らせる測定は 1 ユニットにまとめる。ユニットの中では `~/.cargo/bin` が PATH に無いので、`cargo` を使うコマンドは `export PATH=$HOME/.cargo/bin:$PATH;` から始めるか、`bash -l` のスクリプトにする。
- `runs/*.sh`: 各測定の実際の起動手順。`probe.bin` は、その測定の時点のコミットで `cargo build --release --example <名前>` した実行ファイルの写し。
  - `onset_v10.sh`・`onset_v11.sh`: 声の始まりを 1 ms で測り直したもの（声門の開きを口の解放に合わせるか、ずらし量）。`devoicing_probe`。
  - `frication_v3.sh`: 無声の母音の舌の高さと摩擦の音源。`frication_probe`。
  - `bz_v1.sh`: 子音の後の F0 を Birkholz & Zhang 2020 と同じ読み方で読む。`microprosody_probe --bz`。
  - `baseline_wer.sh`・`baseline_compare.sh`: 本家の main とこのフォークの main の基準（#1）。作業ツリーは `git worktree add /hdd/chimera-wt/<名前> <コミット>` で作る。

## 集計

- `analysis/summ_v6.py`・`analysis/summ_v8.py`: `devoicing_probe` の出力から D50（母音が半分無声になる開きの長さ）を出す。
- `analysis/jvs_cf0.py JVS_ROOT OUT_DIR [WORKERS]`: JVS（parallel100）の人の音声で、子音の後の母音の最初の 80 ms の F0 を Praat（parselmouth）で読み、45〜75 ms の回帰直線を引いた残りを話者ごとに出す。
- `analysis/jvs_cf0_v2_summary.py tokens.tsv`: 上の出力から、母音をそろえて「無声の阻害音 − 共鳴音」の差を、語頭と語中・長母音に分けて話者の間で平均する。

Python は `python3 -I` で、numpy と parselmouth が要る。
