"""Summarise devoicing_probe v8 (the v7 retake): the output measure only.

D50: the shortest open-interval length (ms) from which `voiced`, divided by the vowel-matched
voiced control's `voiced` at the same open-interval length (linear interpolation on the control's
curve), stays at or above one half (last upward crossing, interpolated). Controls: /i/ gida, /a/
gada, /ɯ/ gɯda, at the same glottis, VOT, tongue (adv does not touch them; adv 0 is used).
Usage: python3 summ_v8.py FILE [FILE ...]  (one table per file)
"""
import collections
import csv
import sys

VOWEL_CTRL = {'i': 'gida', 'a': 'gada', 'ɯ': 'gɯda'}


def target_vowel(word):
    for ch in word:
        if ch in 'iaɯeo':
            return ch
    return None


def load(path):
    head = [l for l in open(path) if l.startswith('#')]
    rows = list(csv.DictReader((l for l in open(path) if not l.startswith('#')), delimiter='\t'))
    for r in rows:
        for k in ('adv_ms', 'nominal_ms', 'open_ms', 'ps', 'pg', 'dp', 'abd'):
            r[k] = float(r[k])
        for k in ('voiced', 'lead_ms', 'tail_ms', 'f_close', 'f_swing', 'f_still'):
            r[k] = float(r[k]) if r.get(k) not in (None, '') else None
    return head, rows


def interp(curve, x):
    """curve: sorted (x, y) with y not None."""
    if not curve:
        return None
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
    return None


def main(path):
    head, rows = load(path)
    print(''.join(head), end='')
    key = lambda r: (r['block'], r['word'], r['glottis'], r['vot'], r['adv_ms'], r['larynx'], r['tongue'])
    g = collections.defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    ctrl = {}
    for (blk, w, gl, vot, adv, lar, tg), seq in g.items():
        if blk == 'main' and w in VOWEL_CTRL.values() and adv == 0:
            ctrl[(w, gl, vot, tg)] = sorted((r['open_ms'], r['voiced']) for r in seq if r['voiced'] is not None)

    def norm(r):
        c = ctrl.get((VOWEL_CTRL.get(target_vowel(r['word'])), r['glottis'], r['vot'], r['tongue']))
        if c is None or r['voiced'] is None:
            return None
        cv = interp(c, r['open_ms'])
        if cv is None or cv < 0.2:
            return None
        return r['voiced'] / cv

    def d50(seq):
        pts = [(r['open_ms'], norm(r)) for r in sorted(seq, key=lambda r: r['open_ms'])]
        pts = [p for p in pts if p[1] is not None]
        if not pts:
            return '-'
        if all(v >= 0.5 for _, v in pts):
            return '<%d' % pts[0][0]
        if pts[-1][1] < 0.5:
            return '>%d' % pts[-1][0]
        for i in range(len(pts) - 1, 0, -1):
            if pts[i - 1][1] < 0.5 <= pts[i][1]:
                (a, va), (b, vb) = pts[i - 1], pts[i]
                return '%.0f' % (a + (0.5 - va) * (b - a) / (vb - va))
        return '?'

    words = []
    for r in rows:
        if r['word'] not in words:
            words.append(r['word'])
    gls = sorted({r['glottis'] for r in rows})
    vots = ['unasp', '30ms', '50ms']
    advs = [0.0, 20.0, 40.0, 60.0]
    for gl in gls:
        print(f"\n== {gl}: D50 (open-interval ms) on voiced / vowel-matched control; overlap, tongue target; cells adv 0/20/40/60")
        print('word'.ljust(6), *[v.rjust(22) for v in vots])
        for w in words:
            cells = ['/'.join(d50(g[('main', w, gl, v, a, 'overlap', 'target')]) for a in advs) for v in vots]
            print(w.ljust(6), *[c.rjust(22) for c in cells])
        print(f"\n== {gl}: near 60 ms open interval, unasp, adv 0, target: lead tail (ms), ps pg dp, abd, folds close/swing/still")
        for w in words:
            seq = g[('main', w, gl, 'unasp', 0.0, 'overlap', 'target')]
            if not seq:
                continue
            r = min(seq, key=lambda r: abs(r['open_ms'] - 60))
            f = lambda x, d=1: '-' if x is None else f'{x:.{d}f}'
            print(w.ljust(6), f"{r['open_ms']:.0f}", f(r['lead_ms']), f(r['tail_ms']), f"{r['ps']:.2f} {r['pg']:.2f} {r['dp']:.2f}", f"{r['abd']:.3f}", f(r['f_close'], 2), f(r['f_swing'], 2), f(r['f_still'], 2))
        sizes = sorted({r['larynx'] for r in rows if r['block'] == 'peak' and r['glottis'] == gl})
        if sizes:
            print(f"\n== {gl}: peak, D50 per size, unasp")
            for w in ['kita', 'kɯta', 'kɯɕa']:
                print(w.ljust(6), ' '.join(f"{s[4:]}:{d50(g[('peak', w, gl, 'unasp', 0.0, s, 'target')])}" for s in sizes))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(f"\n######## {p}")
        main(p)
