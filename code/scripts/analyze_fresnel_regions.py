#!/usr/bin/env python3
"""Sample left/right sphere regions in Fresnel Cornell renders."""
from PIL import Image
import numpy as np
import sys

def analyze(path, left_cx_frac=0.35, right_cx_frac=0.65, cy_frac=0.55, r_frac=0.12):
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    ys = np.arange(h)
    xs = np.arange(w)
    xx, yy = np.meshgrid(xs, ys)

    def sphere_stats(cx_frac, label):
        cx, cy = int(w * cx_frac), int(h * cy_frac)
        r = int(min(w, h) * r_frac)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        region = img[mask]
        mean = region.mean(axis=0)
        std = region.std(axis=0)
        lum = 0.2126 * region[:, 0] + 0.7152 * region[:, 1] + 0.0722 * region[:, 2]
        print(f"  {label} @ ({cx},{cy}) r={r}: mean RGB=({mean[0]:.3f},{mean[1]:.3f},{mean[2]:.3f}) "
              f"std=({std[0]:.3f},{std[1]:.3f},{std[2]:.3f}) lum_mean={lum.mean():.3f} lum_std={lum.std():.3f}")
        return mean, lum.std()

    print(f"\n=== {path} ({w}x{h}) ===")
    global_mean = img.mean(axis=(0, 1))
    print(f"  Global mean RGB=({global_mean[0]:.3f},{global_mean[1]:.3f},{global_mean[2]:.3f})")
    left_mean, left_std = sphere_stats(left_cx_frac, "LEFT sphere")
    right_mean, right_std = sphere_stats(right_cx_frac, "RIGHT sphere")
    # rim vs center for right sphere
    cx, cy = int(w * right_cx_frac), int(h * cy_frac)
    r_outer = int(min(w, h) * r_frac)
    r_inner = int(r_outer * 0.55)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    rim = img[(dist2 <= r_outer ** 2) & (dist2 >= r_inner ** 2)]
    center = img[dist2 < r_inner ** 2]
    rim_lum = (0.2126 * rim[:, 0] + 0.7152 * rim[:, 1] + 0.0722 * rim[:, 2]).mean()
    ctr_lum = (0.2126 * center[:, 0] + 0.7152 * center[:, 1] + 0.0722 * center[:, 2]).mean()
    print(f"  RIGHT rim lum={rim_lum:.3f} center lum={ctr_lum:.3f} rim/center={rim_lum/max(ctr_lum,1e-6):.2f}x")
    grayness = abs(left_mean[0] - left_mean[1]) + abs(left_mean[1] - left_mean[2])
    print(f"  LEFT grayness (|R-G|+|G-B|)={grayness:.3f}  RIGHT={abs(right_mean[0]-right_mean[1])+abs(right_mean[1]-right_mean[2]):.3f}")
    left_lum = 0.2126 * left_mean[0] + 0.7152 * left_mean[1] + 0.0722 * left_mean[2]
    right_lum = 0.2126 * right_mean[0] + 0.7152 * right_mean[1] + 0.0722 * right_mean[2]
    print(f"  LEFT lum={left_lum:.3f}  RIGHT lum={right_lum:.3f}  delta={right_lum-left_lum:+.3f}")
    if left_mean[0] > 1e-6:
        print(f"  LEFT B/R={left_mean[2]/left_mean[0]:.3f}  RIGHT B/R={right_mean[2]/max(right_mean[0],1e-6):.3f}")
    # center vs back wall (transparency check)
    cx_l, cy = int(w * left_cx_frac), int(h * cy_frac)
    cx_r = int(w * right_cx_frac)
    bw_y, bw_x = int(h * 0.52), int(w * 0.50)
    backwall = img[bw_y, bw_x]
    left_ctr = img[cy, cx_l]
    right_ctr = img[cy, cx_r]
    print(f"  Back wall ref RGB=({backwall[0]:.3f},{backwall[1]:.3f},{backwall[2]:.3f})")
    print(f"  LEFT center RGB=({left_ctr[0]:.3f},{left_ctr[1]:.3f},{left_ctr[2]:.3f}) "
          f"L2_to_backwall={np.linalg.norm(left_ctr-backwall):.3f}")
    print(f"  RIGHT center RGB=({right_ctr[0]:.3f},{right_ctr[1]:.3f},{right_ctr[2]:.3f}) "
          f"L2_to_backwall={np.linalg.norm(right_ctr-backwall):.3f}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
