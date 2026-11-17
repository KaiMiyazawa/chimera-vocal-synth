"""F0 at the start of vowels after voiceless / voiced obstruents and sonorants in JVS (parallel100),
read the way Birkholz & Zhang 2020 (ICASSP) read it: Praat F0 at eight equidistant points over the
first 80 ms of the vowel (5, 15, ..., 75 ms), averaged over vowels, then minus a regression line fitted
through the 45-75 ms points. The residual at 5-35 ms is the consonant's effect on F0 (CF0).

Differences from B&Z (state them when quoting): Japanese, 100 speakers (not one German male); vowel
onsets come from JVS's automatic alignment (10 ms steps); Praat's pitch range is each speaker's range
from gender_f0range.txt; the average and the line are taken per speaker, and the speakers are the units.
Vowels: a i u e o (devoiced I/U excluded), a run of the same vowel merged (long vowel), at least 80 ms
long, directly after the consonant.

Usage: python3 -I jvs_cf0.py JVS_ROOT OUT_DIR [WORKERS]
"""
import collections
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import parselmouth

POINTS = [0.005 + 0.010 * k for k in range(8)]  # 5..75 ms
FIT = [4, 5, 6, 7]  # 45..75 ms
VOICELESS = {'p', 't', 'k', 'ky', 'py', 'ch', 'ts', 's', 'sh', 'h', 'hy', 'f'}
VOICED = {'b', 'd', 'g', 'gy', 'by', 'z', 'j'}
SONORANT = {'m', 'n', 'my', 'ny', 'r', 'ry', 'w', 'y'}
VOWELS = set('aiueo')


def klass(c):
    if c in VOICELESS:
        return 'voiceless'
    if c in VOICED:
        return 'voiced'
    if c in SONORANT:
        return 'sonorant'
    return None


def read_lab(path):
    segs = []
    for line in open(path):
        p = line.split()
        if len(p) == 3:
            segs.append((float(p[0]), float(p[1]), p[2]))
    return segs


def tokens(segs):
    """(consonant, vowel, onset_s, dur_s) for vowels directly after a consonant."""
    out = []
    i = 1
    while i < len(segs):
        s0, e0, ph = segs[i]
        if ph in VOWELS:
            j = i
            while j + 1 < len(segs) and segs[j + 1][2] == ph:
                j += 1
            c = segs[i - 1][2]
            if klass(c):
                before = segs[i - 2][2] if i >= 2 else 'sil'
                pos = 'initial' if before in ('sil', 'pau') else 'medial'
                out.append((c, ph, s0, segs[j][1] - s0, pos, int(j > i)))
            i = j + 1
        else:
            i += 1
    return out


def one_file(args):
    wav, lab, lo, hi = args
    try:
        snd = parselmouth.Sound(wav)
        pitch = snd.to_pitch(time_step=0.005, pitch_floor=lo, pitch_ceiling=hi)
    except Exception as e:  # noqa: BLE001 - report and keep going
        return wav, None, repr(e)
    rows = []
    for c, v, t0, dur, pos, long_ in tokens(read_lab(lab)):
        f = [pitch.get_value_at_time(t0 + p) for p in POINTS]
        rows.append((c, v, dur, pos, long_, [x if x == x else math.nan for x in f]))
    return wav, rows, None


def detrend(mean):
    x = np.array([POINTS[k] for k in FIT])
    y = np.array([mean[k] for k in FIT])
    if np.isnan(y).any():
        return [math.nan] * 8
    b, a = np.polyfit(x, y, 1)
    return [mean[k] - (a + b * POINTS[k]) for k in range(8)]


def main():
    root, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(out, exist_ok=True)
    spk = {}
    for line in open(os.path.join(root, 'gender_f0range.txt')):
        p = line.split()
        if p and p[0].startswith('jvs'):
            spk[p[0]] = (p[1], float(p[2]), float(p[3]))
    jobs = []
    for s, (_, lo, hi) in sorted(spk.items()):
        d = os.path.join(root, s, 'parallel100')
        labdir = os.path.join(d, 'lab', 'mon')
        if not os.path.isdir(labdir):
            continue
        for f in sorted(os.listdir(labdir)):
            wav = os.path.join(d, 'wav24kHz16bit', f[:-4] + '.wav')
            if os.path.exists(wav):
                jobs.append((wav, os.path.join(labdir, f), lo, hi))
    errs = 0
    # per speaker, per (class or consonant), per min-duration: list of contours
    acc = collections.defaultdict(list)
    with open(os.path.join(out, 'tokens.tsv'), 'w') as tf, mp.Pool(workers) as pool:
        tf.write('speaker\tgender\tfile\tcons\tclass\tvowel\tdur_ms\tpos\tlong\t' + '\t'.join(f'f{round(p * 1000)}' for p in POINTS) + '\n')
        for wav, rows, err in pool.imap_unordered(one_file, jobs, chunksize=8):
            s = wav.split(os.sep)[-4]
            if err:
                errs += 1
                print('ERR', wav, err, file=sys.stderr)
                continue
            for c, v, dur, pos, long_, f in rows:
                k = klass(c)
                tf.write(f"{s}\t{spk[s][0]}\t{os.path.basename(wav)}\t{c}\t{k}\t{v}\t{dur * 1000:.0f}\t{pos}\t{long_}\t" + '\t'.join('nan' if math.isnan(x) else f'{x:.2f}' for x in f) + '\n')
                if dur >= 0.080:
                    acc[(s, k)].append(f)
                    if k == 'voiceless':
                        acc[(s, c)].append(f)
    print(f'files {len(jobs)}  errors {errs}')

    def per_speaker(key, semitone):
        f = np.array(acc[key], dtype=float)
        if len(f) < 10:
            return None
        if semitone:
            ref = np.nanmean(f)
            f = 12 * np.log2(f / ref)
        m = np.nanmean(f, axis=0)
        return detrend(list(m)), len(f), float(np.mean(~np.isnan(f)))

    labels = ['voiceless', 'voiced', 'sonorant'] + sorted(VOICELESS)
    with open(os.path.join(out, 'speakers.tsv'), 'w') as sf:
        sf.write('speaker\tgender\tgroup\tunit\tn\tvoiced_frac\t' + '\t'.join(f'r{int(p * 1000)}' for p in POINTS) + '\n')
        summary = collections.defaultdict(list)
        for s in sorted(spk):
            g = spk[s][0]
            for lab in labels:
                for unit, st in (('Hz', False), ('st', True)):
                    r = per_speaker((s, lab), st)
                    if r is None:
                        continue
                    res, n, vf = r
                    sf.write(f'{s}\t{g}\t{lab}\t{unit}\t{n}\t{vf:.2f}\t' + '\t'.join(f'{x:.3f}' for x in res) + '\n')
                    summary[(lab, unit, g)].append(res)
                    summary[(lab, unit, 'all')].append(res)
    for unit in ('Hz', 'st'):
        print(f'\n== residual from the 45-75 ms line ({unit}); mean over speakers [95% CI], points 5/15/25/35 ms; n speakers; speakers with r5>0')
        for lab in labels:
            for g in ('M', 'F', 'all'):
                rs = np.array(summary.get((lab, unit, g), []))
                if len(rs) == 0:
                    continue
                cells = []
                for k in range(4):
                    col = rs[:, k][~np.isnan(rs[:, k])]
                    if len(col) < 2:
                        cells.append('-')
                        continue
                    m, se = col.mean(), col.std(ddof=1) / math.sqrt(len(col))
                    cells.append(f'{m:+.2f}[{m - 1.96 * se:+.2f},{m + 1.96 * se:+.2f}]')
                pos = int(np.sum(rs[:, 0] > 0))
                print(f'{lab:10s} {g:3s} ' + '  '.join(c.rjust(22) for c in cells) + f'  n={len(rs)}  r5>0: {pos}')


if __name__ == '__main__':
    main()
