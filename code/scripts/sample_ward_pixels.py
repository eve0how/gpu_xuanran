#!/usr/bin/env python3
"""Sample top 20% of Ward sphere pixels from rendered BMP."""
import struct
import sys

def read_bmp(path):
    with open(path, 'rb') as f:
        if f.read(2) != b'BM':
            raise ValueError('not BMP')
        f.read(8)  # file size + reserved
        offset = struct.unpack('<I', f.read(4))[0]
        dib = struct.unpack('<I', f.read(4))[0]
        if dib >= 40:
            w, h_raw = struct.unpack('<ii', f.read(8))
            planes, bpp = struct.unpack('<HH', f.read(4))
        else:
            raise ValueError(f'unsupported dib {dib}')
        h = abs(h_raw)
        top_down = h_raw < 0
        f.seek(offset)
        row_bytes = ((w * (bpp // 8) + 3) // 4) * 4
        rows = []
        for _ in range(h):
            row = f.read(row_bytes)
            row_px = []
            for x in range(w):
                b, g, r = row[x*3], row[x*3+1], row[x*3+2]
                row_px.append((r, g, b))
            rows.append(row_px)
        pixels = rows if top_down else rows[::-1]
    return w, h, pixels

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'output/ward/ward_aniso_demo.bmp'
    w, h, px = read_bmp(path)
    print(f'size {w}x{h}')

    # Bronze-ish mask in lower half
    bronze = []
    for y in range(h // 2, h):
        for x in range(w):
            r, g, b = px[y][x]
            if r > 80 and g < 130 and b < 100 and r > b:
                bronze.append((x, y, r, g, b))

    mid = w // 2
    for name, filt in [('LEFT', lambda p: p[0] < mid), ('RIGHT', lambda p: p[0] >= mid)]:
        pts = [p for p in bronze if filt(p)]
        if not pts:
            print(f'{name}: no pixels')
            continue
        ys = [p[1] for p in pts]
        ymin, ymax = min(ys), max(ys)
        thr = ymin + 0.2 * (ymax - ymin)
        top = [p for p in pts if p[1] <= thr]
        rs = [p[2] for p in top]
        gs = [p[3] for p in top]
        bs = [p[4] for p in top]
        print(f'\n{name} n={len(top)} y=[{ymin},{ymax}] top20% y<={thr:.0f}')
        print(f'  max RGB: ({max(rs)}, {max(gs)}, {max(bs)})')
        print(f'  mean RGB: ({sum(rs)//len(rs)}, {sum(gs)//len(gs)}, {sum(bs)//len(bs)})')
        white = sum(1 for p in top if p[2] >= 254 and p[3] >= 254 and p[4] >= 254)
        print(f'  white254: {white}/{len(top)} ({100*white/len(top):.1f}%)')
        for p in sorted(top, key=lambda q: q[1])[:5]:
            print(f'  ({p[0]},{p[1]}): {p[2:]}')

if __name__ == '__main__':
    main()
