#!/usr/bin/env python3
from PIL import Image
import numpy as np

paths = [
    ("compare", "/data/PA1-2/code/output/fresnel/fresnel_cornell_compare.bmp"),
    ("water_glass", "/data/PA1-2/code/output/fresnel/fresnel_cornell_water_glass.bmp"),
    ("grazing", "/data/PA1-2/code/output/fresnel/fresnel_cornell_grazing.bmp"),
]

for name, path in paths:
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    # quadrant analysis
    quads = {
        "TL": img[:h//2, :w//2],
        "TR": img[:h//2, w//2:],
        "BL": img[h//2:, :w//2],
        "BR": img[h//2:, w//2:],
    }
    print(f"\n{name}:")
    for q, r in quads.items():
        m = r.mean(axis=(0,1))
        print(f"  {q}: R={m[0]:.3f} G={m[1]:.3f} B={m[2]:.3f}")
    # dominant hue row slices at y=512
    row = img[h//2]
    left = row[:w//3].mean(axis=0)
    mid = row[w//3:2*w//3].mean(axis=0)
    right = row[2*w//3:].mean(axis=0)
    print(f"  mid-row L/M/R: {left} / {mid} / {right}")
