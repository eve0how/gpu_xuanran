#!/usr/bin/env python3
"""Compare two BMP images pixel-by-pixel."""
import struct
import sys
import math


def read_bmp(path):
    with open(path, "rb") as f:
        header = f.read(14)
        if header[:2] != b"BM":
            raise ValueError(f"{path}: not a BMP")
        dib = f.read(40)
        width, height = struct.unpack("<ii", dib[4:12])
        bpp = struct.unpack("<H", dib[14:16])[0]
        if bpp != 24:
            raise ValueError(f"{path}: expected 24bpp, got {bpp}")
        abs_h = abs(height)
        row_bytes = ((bpp * width + 31) // 32) * 4
        pixels = []
        for y in range(abs_h):
            row = f.read(row_bytes)
            for x in range(width):
                b, g, r = row[x * 3 : x * 3 + 3]
                pixels.append((r, g, b))
        if height > 0:
            pixels = pixels[::-1]
        return width, abs_h, pixels


def compare(path_a, path_b, threshold=5):
    w1, h1, px1 = read_bmp(path_a)
    w2, h2, px2 = read_bmp(path_b)
    if (w1, h1) != (w2, h2):
        print(f"SIZE MISMATCH: {path_a} {w1}x{h1} vs {path_b} {w2}x{h2}")
        return 1

    n = w1 * h1
    diffs = []
    max_diff = 0
    sum_diff = 0.0
    over = 0
    for i in range(n):
        d = sum(abs(px1[i][c] - px2[i][c]) for c in range(3))
        diffs.append(d)
        max_diff = max(max_diff, d)
        sum_diff += d
        if d > threshold:
            over += 1

    mean_diff = sum_diff / n
    rms = math.sqrt(sum(d * d for d in diffs) / n)
    print(f"Compare: {path_a} vs {path_b}")
    print(f"  Size: {w1}x{h1}, pixels: {n}")
    print(f"  Mean L1 diff: {mean_diff:.2f}")
    print(f"  RMS L1 diff: {rms:.2f}")
    print(f"  Max L1 diff: {max_diff}")
    print(f"  Pixels > {threshold}: {over} ({100.0 * over / n:.2f}%)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_bmp.py <a.bmp> <b.bmp>")
        sys.exit(1)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
