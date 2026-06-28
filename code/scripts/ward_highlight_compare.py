#!/usr/bin/env python3
"""Measure Ward sphere highlight shape: aspect ratio and zoom crops."""
import struct
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


def read_bmp(path):
    with open(path, 'rb') as f:
        if f.read(2) != b'BM':
            raise ValueError('not BMP')
        f.read(8)
        offset = struct.unpack('<I', f.read(4))[0]
        dib = struct.unpack('<I', f.read(4))[0]
        w, h_raw = struct.unpack('<ii', f.read(8))
        planes, bpp = struct.unpack('<HH', f.read(4))
        h = abs(h_raw)
        top_down = h_raw < 0
        f.seek(offset)
        row_bytes = ((w * (bpp // 8) + 3) // 4) * 4
        rows = []
        for _ in range(h):
            row = f.read(row_bytes)
            row_px = []
            for x in range(w):
                b, g, r = row[x * 3], row[x * 3 + 1], row[x * 3 + 2]
                row_px.append((r, g, b))
            rows.append(row_px)
        pixels = rows if top_down else rows[::-1]
    return w, h, pixels


def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def sphere_highlight_stats(px, w, h, side, lum_thr=95):
    mid = w // 2
    x0, x1 = (0, mid) if side == 'LEFT' else (mid, w)
    bronze = []
    for y in range(h // 2, h):
        for x in range(x0, x1):
            r, g, b = px[y][x]
            if r > 70 and g < 140 and b < 110 and r > b:
                bronze.append((x, y, luminance((r, g, b))))
    if not bronze:
        return None
    ys = [p[1] for p in bronze]
    ymin, ymax = min(ys), max(ys)
    top_band = [p for p in bronze if p[1] <= ymin + 0.35 * (ymax - ymin)]
    bright = [p for p in top_band if p[2] >= lum_thr]
    if len(bright) < 8:
        bright = sorted(top_band, key=lambda q: q[2], reverse=True)[:max(8, len(top_band) // 5)]
    xs = [p[0] for p in bright]
    ys2 = [p[1] for p in bright]
    xmin, xmax = min(xs), max(xs)
    ymin2, ymax2 = min(ys2), max(ys2)
    width = max(1, xmax - xmin + 1)
    height = max(1, ymax2 - ymin2 + 1)
    aspect = max(width, height) / min(width, height)
    return {
        'side': side,
        'n': len(bright),
        'bbox': (xmin, ymin2, xmax, ymax2),
        'width': width,
        'height': height,
        'aspect': aspect,
        'max_lum': max(p[2] for p in bright),
    }


def save_crops(path, px, w, h, out_dir):
    if Image is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    mid = w // 2
    for side, x0, x1 in [('left', 0, mid), ('right', mid, w)]:
        stats = sphere_highlight_stats(px, w, h, side.upper())
        if not stats:
            continue
        xmin, ymin, xmax, ymax = stats['bbox']
        pad = 24
        cx0 = max(x0, xmin - pad)
        cy0 = max(0, ymin - pad)
        cx1 = min(x1, xmax + pad)
        cy1 = min(h, ymax + pad)
        crop = Image.new('RGB', (cx1 - cx0, cy1 - cy0))
        pix = crop.load()
        for y in range(cy0, cy1):
            for x in range(cx0, cx1):
                pix[x - cx0, y - cy0] = px[y][x]
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST)
        out = out_dir / f'{stem}_{side}_highlight.png'
        crop.save(out)
        print(f'  crop -> {out}')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'output/ward/ward_aniso_demo.bmp'
    w, h, px = read_bmp(path)
    print(f'{path}: {w}x{h}')
    for side in ('LEFT', 'RIGHT'):
        s = sphere_highlight_stats(px, w, h, side)
        if not s:
            print(f'{side}: no highlight region')
            continue
        print(f"\n{side}: n={s['n']} bbox={s['bbox']} size={s['width']}x{s['height']} "
              f"aspect={s['aspect']:.2f}:1 max_lum={s['max_lum']:.0f}")
    if len(sys.argv) > 2:
        save_crops(path, px, w, h, sys.argv[2])
    elif Image is not None:
        save_crops(path, px, w, h, 'output/ward')


if __name__ == '__main__':
    main()
