"""Summarise devoicing_probe v6: the open-interval length at which a vowel stops being mostly voiced.

D50 here: the shortest open-interval length (ms) from which the measure stays at or above half of
the voiced control's value at the same nominal length (normalised), read by linear interpolation
on the last upward crossing. Both the output measure (voiced) and the model's gate are reported.
"""
import collections
import csv
import sys

path = sys.argv[1]
rows = [r for r in csv.DictReader((l for l in open(path) if not l.startswith('#')), delimiter='\t')]
for r in rows:
    for k in ('adv_ms', 'nominal_ms', 'open_ms', 'gate', 'p_gate', 'ps', 'pg', 'dp', 'abd', 'tb_y'):
        r[k] = float(r[k])
    r['voiced'] = float(r['voiced']) if r['voiced'] else None

key = lambda r: (r['block'], r['word'], r['glottis'], r['vot'], r['adv_ms'], r['larynx'], r['tongue'])
g = collections.defaultdict(list)
for r in rows:
    g[key(r)].append(r)

# the voiced control gida at the same glottis model and nominal length (VOT and adv do not touch it)
ctrl = {}
for r in rows:
    if r['word'] == 'gida' and r['block'] == 'main' and r['tongue'] == 'target' and r['adv_ms'] == 0 and r['vot'] == 'unasp':
        ctrl[(r['glottis'], r['nominal_ms'])] = r


def norm(r, m):
    c = ctrl[(r['glottis'], r['nominal_ms'])]
    v, cv = r[m], c[m]
    if v is None or cv is None or cv < 0.2:
        return None
    return v / cv


def d50(seq, m, normalise=True):
    pts = []
    for r in sorted(seq, key=lambda r: r['open_ms']):
        v = norm(r, m) if normalise else r[m]
        if v is not None:
            pts.append((r['open_ms'], v))
    if not pts:
        return '-'
    if all(v >= 0.5 for _, v in pts):
        return '<%d' % pts[0][0]
    if pts[-1][1] < 0.5:
        return '>%d' % pts[-1][0]
    for i in range(len(pts) - 1, 0, -1):
        if pts[i - 1][1] < 0.5 <= pts[i][1]:
            (d0, v0), (d1, v1) = pts[i - 1], pts[i]
            return '%.0f' % (d0 + (0.5 - v0) * (d1 - d0) / (v1 - v0))
    return '?'


words = []
for r in rows:
    if r['word'] not in words:
        words.append(r['word'])
VOT = ['unasp', '30ms', '50ms']
ADV = [0.0, 20.0, 40.0, 60.0]

for gl in ['cutoff', 'threshold']:
    for m in ['voiced', 'gate']:
        print(f"\n== glottis {gl}, D50 on {m} (normalised by gida), open-interval ms; overlap, tongue target; cells adv 0/20/40/60 ms")
        print('word'.ljust(6), *[v.rjust(24) for v in VOT])
        for w in words:
            cells = ['/'.join(d50(g[('main', w, gl, v, a, 'overlap', 'target')], m) for a in ADV) for v in VOT]
            print(w.ljust(6), *[c.rjust(24) for c in cells])

print("\n== kita minus kida (D50 on gate, normalised): positive = kita devoices at longer vowels")
for gl in ['cutoff', 'threshold']:
    for v in VOT:
        out = []
        for a in ADV:
            x, y = d50(g[('main', 'kita', gl, v, a, 'overlap', 'target')], 'gate'), d50(g[('main', 'kida', gl, v, a, 'overlap', 'target')], 'gate')
            try:
                out.append('%+d' % (int(x) - int(y)))
            except ValueError:
                out.append(f'{x}|{y}')
        print(gl.ljust(10), v.ljust(6), ' '.join(o.rjust(10) for o in out))

print("\n== tongue none vs target (D50 on gate, normalised), adv 0, unasp")
for gl in ['cutoff', 'threshold']:
    for w in words:
        a, b = d50(g[('main', w, gl, 'unasp', 0.0, 'overlap', 'target')], 'gate'), d50(g[('main', w, gl, 'unasp', 0.0, 'overlap', 'none')], 'gate')
        print(gl.ljust(10), w.ljust(6), f'{a}/{b}')

print("\n== peak: D50 on gate (normalised) per peak size, unasp; cells cutoff/threshold")
sizes = sorted({r['larynx'] for r in rows if r['block'] == 'peak'})
for w in ['kita', 'kɯta', 'kɯɕa']:
    print(w.ljust(6), ' '.join(f"{s[4:]}:{d50(g[('peak', w, 'cutoff', 'unasp', 0.0, s, 'target')], 'gate')}/{d50(g[('peak', w, 'threshold', 'unasp', 0.0, s, 'target')], 'gate')}" for s in sizes))

print("\n== pressures at ~60 ms open interval, cutoff, overlap, unasp, adv 0, target: ps pg dp gate abd")
for w in words:
    seq = g[('main', w, 'cutoff', 'unasp', 0.0, 'overlap', 'target')]
    r = min(seq, key=lambda r: abs(r['open_ms'] - 60))
    print(w.ljust(6), '%3.0f' % r['open_ms'], *['%.2f' % r[k] for k in ('ps', 'pg', 'dp', 'gate', 'abd')])
