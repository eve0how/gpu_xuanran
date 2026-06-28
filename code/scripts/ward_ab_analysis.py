#!/usr/bin/env python3
"""Compare Ward aniso A vs B highlight streak orientation."""
import struct
import sys
from pathlib import Path


def read_bmp(path):
    with open(path, 'rb') as f:
        if f.read(2) != b'BM':
            raise ValueError('not BMP')
        f.read(8)
        offset = struct.unpack('<I', f.read(4))[0]
        f.read(4)
        w, h = struct.unpack('<ii', f.read(8))
        h = abs(h)
        f.read(8)
        f.seek(offset)
        row = ((w * 3 + 3) // 4) * 4
        px = []
        for _ in range(h):
            r = f.read(row)
            px.append([(r[x * 3 + 2], r[x * 3 + 1], r[x * 3]) for x in range(w)])
        return w, h, px[::-1]


def streak_stats(px, w, h, x0=0, x1=None):
    if x1 is None:
        x1 = w
    best = 0
    bx = by = 0
    for y in range(h):
        for x in range(x0, x1):
            r, g, b = px[y][x]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if lum > best:
                best, bx, by = lum, x, y
    hline = [px[by][x] for x in range(max(x0, bx - 100), min(x1, bx + 100))]
    vline = [px[y][bx] for y in range(max(0, by - 100), min(h, by + 100))]

    def width(line):
        peak = max(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for p in line)
        thr = peak * 0.5
        idx = [i for i, p in enumerate(line) if 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] >= thr]
        return len(idx) if idx else 0

    hw, vw = width(hline), width(vline)
    orient = 'horizontal' if hw > vw * 1.25 else ('vertical' if vw > hw * 1.25 else 'round')
    return {'peak': (bx, by), 'h_width': hw, 'v_width': vw, 'aspect': max(hw, vw) / max(1, min(hw, vw)), 'orient': orient}


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else [
        'output/ward/ward_aniso_A.bmp',
        'output/ward/ward_aniso_B.bmp',
    ]
    results = []
    for p in paths:
        w, h, px = read_bmp(p)
        s = streak_stats(px, w, h)
        print(f"\n{p}: peak@{s['peak']} H={s['h_width']} V={s['v_width']} aspect={s['aspect']:.2f} ({s['orient']})")
        results.append(s)
    if len(results) == 2:
        a, b = results
        print(f"\nA vs B: orient {a['orient']} -> {b['orient']}")
        if a['orient'] != b['orient']:
            print('PASS: streak direction differs between A and B')
        else:
            print('WARN: streak orientations match — check tangent/alpha setup')


if __name__ == '__main__':
    main()
