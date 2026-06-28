#!/usr/bin/env python3
"""Analyze Ward anisotropy showcase: 3-ball highlight bbox aspect ratios + 3x zoom crops."""
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


def gold_mask(px, w, h, x0, x1):
    pts = []
    y_start = h // 3
    for y in range(y_start, h):
        for x in range(x0, x1):
            r, g, b = px[y][x]
            if r > 60 and g > 40 and b < 120 and r > b and g > b * 0.6:
                pts.append((x, y, luminance((r, g, b))))
    return pts


def highlight_stats(px, w, h, x0, x1, lum_frac=0.72):
    gold = gold_mask(px, w, h, x0, x1)
    if not gold:
        return None
    ys = [p[1] for p in gold]
    ymin, ymax = min(ys), max(ys)
    mid_band = [p for p in gold if ymin + 0.15 * (ymax - ymin) <= p[1] <= ymin + 0.55 * (ymax - ymin)]
    if not mid_band:
        mid_band = gold
    peak = max(p[2] for p in mid_band)
    lum_thr = max(55.0, peak * lum_frac)
    bright = [p for p in mid_band if p[2] >= lum_thr]
    if len(bright) < 6:
        bright = sorted(mid_band, key=lambda q: q[2], reverse=True)[:max(6, len(mid_band) // 8)]
    if not bright:
        return None
    xs = [p[0] for p in bright]
    ys2 = [p[1] for p in bright]
    xmin, xmax = min(xs), max(xs)
    ymin2, ymax2 = min(ys2), max(ys2)
    width = max(1, xmax - xmin + 1)
    height = max(1, ymax2 - ymin2 + 1)
    aspect = max(width, height) / min(width, height)
    orient = 'wide' if width > height * 1.25 else ('tall' if height > width * 1.25 else 'round')
    return {
        'n': len(bright),
        'bbox': (xmin, ymin2, xmax, ymax2),
        'width': width,
        'height': height,
        'aspect': aspect,
        'orient': orient,
        'max_lum': max(p[2] for p in bright),
    }


def thirds(w):
    t = w // 3
    return [('LEFT', 0, t), ('CENTER', t, 2 * t), ('RIGHT', 2 * t, w)]


def halves(w):
    mid = w // 2
    return [('LEFT', 0, mid), ('RIGHT', mid, w)]


def save_crop(px, w, h, x0, x1, stats, out_path, scale=3):
    if Image is None or not stats:
        return
    xmin, ymin, xmax, ymax = stats['bbox']
    pad = 28
    cx0 = max(x0, xmin - pad)
    cy0 = max(0, ymin - pad)
    cx1 = min(x1, xmax + pad)
    cy1 = min(h, ymax + pad)
    crop = Image.new('RGB', (cx1 - cx0, cy1 - cy0))
    pix = crop.load()
    for y in range(cy0, cy1):
        for x in range(cx0, cx1):
            pix[x - cx0, y - cy0] = px[y][x]
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
    crop.save(out_path)


def analyze(path, out_dir='output/ward', n_balls=3):
    w, h, px = read_bmp(path)
    print(f'{path}: {w}x{h}')
    regions = thirds(w) if n_balls == 3 else halves(w)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    results = []
    for name, x0, x1 in regions:
        s = highlight_stats(px, w, h, x0, x1)
        if not s:
            print(f'\n{name}: no highlight region')
            continue
        print(f"\n{name}: n={s['n']} bbox={s['bbox']} size={s['width']}x{s['height']} "
              f"aspect={s['aspect']:.2f}:1 ({s['orient']}) max_lum={s['max_lum']:.0f}")
        out = out_dir / f'{stem}_{name.lower()}_highlight.png'
        save_crop(px, w, h, x0, x1, s, out)
        print(f'  crop -> {out}')
        results.append((name, s))
    return results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'output/ward/ward_aniso_showcase.bmp'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/ward'
    n_balls = 3 if 'showcase' in Path(path).stem and 'rotate' not in Path(path).stem else 2
    analyze(path, out_dir, n_balls)


if __name__ == '__main__':
    main()
