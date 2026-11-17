"""Vowel-matched CF0 contrasts from jvs_cf0.py's tokens.tsv (v2 columns: pos, long).

Per speaker, per vowel and condition: mean contour of each consonant group (vowels >= 80 ms), each minus
its own 45-75 ms regression line (B&Z), then the contrast group - sonorant at the same vowel. Contrasts are
averaged over the vowels that have >= MIN tokens in both groups, then over speakers (mean, 95% CI).
Units: Hz, and semitones re the speaker's mean F0 over all tokens.
Usage: python3 -I jvs_cf0_v2_summary.py tokens.tsv
"""
import collections
import csv
import math
import sys

import numpy as np

P = np.array([5, 15, 25, 35, 45, 55, 65, 75]) / 1000
MIN = 5


def resid(m):
    b, a = np.polyfit(P[4:], m[4:], 1)
    return m - (a + b * P)


def main(path):
    rd = csv.reader(open(path), delimiter='\t')
    head = next(rd)
    ix = {h: i for i, h in enumerate(head)}
    f0 = ix['f5']
    tok = collections.defaultdict(list)
    spk_all = collections.defaultdict(list)
    gender = {}
    for r in rd:
        f = np.array([float(x) for x in r[f0:f0 + 8]])
        s = r[ix['speaker']]
        gender[s] = r[ix['gender']]
        spk_all[s].append(np.nanmean(f))
        if int(r[ix['dur_ms']]) < 80:
            continue
        key = (s, r[ix['pos']], r[ix['long']], r[ix['vowel']])
        tok[key + (r[ix['class']],)].append(f)
        if r[ix['class']] == 'voiceless':
            tok[key + ('c:' + r[ix['cons']],)].append(f)
    ref = {s: np.nanmean(v) for s, v in spk_all.items()}
    groups = ['voiceless', 'voiced', 'c:k', 'c:t', 'c:s', 'c:p', 'c:sh', 'c:ch', 'c:h', 'c:f', 'c:ky', 'c:hy']
    conds = [('initial', '0'), ('medial', '0'), ('medial', '1'), ('initial', '1')]
    for vset_name, vset in (('a e o', 'aeo'), ('i u', 'iu')):
        for unit in ('Hz', 'st'):
            print(f'\n== vowels {vset_name}; contrast vs sonorant, same vowel ({unit}); mean over speakers [95% CI] at 5/15/25/35 ms; n speakers')
            for pos, lg in conds:
                for g in groups:
                    per_spk = []
                    for s in sorted(ref):
                        cs = []
                        for v in vset:
                            a = tok.get((s, pos, lg, v, g), [])
                            b = tok.get((s, pos, lg, v, 'sonorant'), [])
                            if len(a) < MIN or len(b) < MIN:
                                continue
                            A, B = np.array(a), np.array(b)
                            if unit == 'st':
                                A, B = 12 * np.log2(A / ref[s]), 12 * np.log2(B / ref[s])
                            ma, mb = np.nanmean(A, axis=0), np.nanmean(B, axis=0)
                            if np.isnan(ma[4:]).any() or np.isnan(mb[4:]).any():
                                continue
                            cs.append(resid(ma) - resid(mb))
                        if cs:
                            per_spk.append(np.nanmean(cs, axis=0))
                    if len(per_spk) < 10:
                        continue
                    X = np.array(per_spk)
                    cells = []
                    for k in range(4):
                        col = X[:, k][~np.isnan(X[:, k])]
                        m, se = col.mean(), col.std(ddof=1) / math.sqrt(len(col))
                        cells.append(f'{m:+.2f}[{m - 1.96 * se:+.2f},{m + 1.96 * se:+.2f}]')
                    lab = f"{pos}{'-long' if lg == '1' else ''}"
                    print(f'{lab:13s} {g:10s} ' + ' '.join(c.rjust(21) for c in cells) + f'  n={len(X)}')


if __name__ == '__main__':
    main(sys.argv[1])
