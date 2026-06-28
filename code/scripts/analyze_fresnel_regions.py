#!/usr/bin/env python3
"""Sample regions in Fresnel Cornell and polished-floor grazing renders."""
from PIL import Image
import numpy as np
import sys


def find_sphere_centroids(img, r_frac=0.12):
    """Estimate left/right sphere disk centers from luminance blobs."""
    h, w = img.shape[:2]
    lum = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2])
    r = int(min(w, h) * r_frac)
    best = []
    for x0, x1 in ((0, w // 2), (w // 2, w)):
        band = lum[:, x0:x1]
        thr = band.mean() + 0.08 * band.std()
        mask = band > thr
        ys, xs = np.where(mask)
        if len(xs) == 0:
            cx = int((x0 + x1) * 0.5)
            cy = h // 2
        else:
            cx = int(x0 + xs.mean())
            cy = int(ys.mean())
        best.append((cx, cy, r))
    return best


def analyze_single_sphere(path, cx_frac=0.5, cy_frac=0.5, r_frac=0.12):
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    cx, cy = int(w * cx_frac), int(h * cy_frac)
    r = int(min(w, h) * r_frac)
    ys, xs = np.arange(h), np.arange(w)
    xx, yy = np.meshgrid(xs, ys)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    bw_y, bw_x = int(h * 0.52), w // 2
    backwall = img[bw_y, bw_x]
    region = img[mask]
    dists = np.linalg.norm(region - backwall, axis=1)
    min_i = int(np.argmin(dists))
    coords = np.argwhere(mask)
    ty, tx = coords[min_i]
    print(f"\n=== {path} ({w}x{h}) [single sphere] ===")
    print(f"  Back wall ref RGB=({backwall[0]:.3f},{backwall[1]:.3f},{backwall[2]:.3f})")
    print(f"  Center RGB=({img[cy,cx,0]:.3f},{img[cy,cx,1]:.3f},{img[cy,cx,2]:.3f}) "
          f"L2={np.linalg.norm(img[cy,cx]-backwall):.3f}")
    print(f"  Best-transmit @ ({tx},{ty}) RGB=({region[min_i,0]:.3f},{region[min_i,1]:.3f},"
          f"{region[min_i,2]:.3f}) L2={dists[min_i]:.3f}")


def analyze_spheres(path, left_cx_frac=None, right_cx_frac=None, cy_frac=None, r_frac=0.12):
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    ys = np.arange(h)
    xs = np.arange(w)
    xx, yy = np.meshgrid(xs, ys)

    if left_cx_frac is None or right_cx_frac is None or cy_frac is None:
        (cx_l, cy_l, r), (cx_r, cy_r, _) = find_sphere_centroids(img, r_frac)
    else:
        cx_l, cy_l = int(w * left_cx_frac), int(h * cy_frac)
        cx_r, cy_r = int(w * right_cx_frac), int(h * cy_frac)
        r = int(min(w, h) * r_frac)

    def sphere_stats(cx, cy, label):
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        region = img[mask]
        mean = region.mean(axis=0)
        std = region.std(axis=0)
        lum = 0.2126 * region[:, 0] + 0.7152 * region[:, 1] + 0.0722 * region[:, 2]
        print(f"  {label} @ ({cx},{cy}) r={r}: mean RGB=({mean[0]:.3f},{mean[1]:.3f},{mean[2]:.3f}) "
              f"std=({std[0]:.3f},{std[1]:.3f},{std[2]:.3f}) lum_mean={lum.mean():.3f} lum_std={lum.std():.3f}")
        return mean, mask

    print(f"\n=== {path} ({w}x{h}) ===")
    global_mean = img.mean(axis=(0, 1))
    print(f"  Global mean RGB=({global_mean[0]:.3f},{global_mean[1]:.3f},{global_mean[2]:.3f})")
    left_mean, left_mask = sphere_stats(cx_l, cy_l, "LEFT sphere")
    right_mean, right_mask = sphere_stats(cx_r, cy_r, "RIGHT sphere")

    r_outer = r
    r_inner = int(r_outer * 0.55)
    dist2_r = (xx - cx_r) ** 2 + (yy - cy_r) ** 2
    rim = img[(dist2_r <= r_outer ** 2) & (dist2_r >= r_inner ** 2)]
    center = img[dist2_r < r_inner ** 2]
    rim_lum = (0.2126 * rim[:, 0] + 0.7152 * rim[:, 1] + 0.0722 * rim[:, 2]).mean()
    ctr_lum = (0.2126 * center[:, 0] + 0.7152 * center[:, 1] + 0.0722 * center[:, 2]).mean()
    print(f"  RIGHT rim lum={rim_lum:.3f} center lum={ctr_lum:.3f} rim/center={rim_lum/max(ctr_lum,1e-6):.2f}x")
    left_lum = 0.2126 * left_mean[0] + 0.7152 * left_mean[1] + 0.0722 * left_mean[2]
    right_lum = 0.2126 * right_mean[0] + 0.7152 * right_mean[1] + 0.0722 * right_mean[2]
    print(f"  LEFT lum={left_lum:.3f}  RIGHT lum={right_lum:.3f}  delta={right_lum-left_lum:+.3f}")
    if left_mean[0] > 1e-6:
        print(f"  LEFT B/R={left_mean[2]/left_mean[0]:.3f}  RIGHT B/R={right_mean[2]/max(right_mean[0],1e-6):.3f}")

    bw_y, bw_x = int(h * 0.52), int(w * 0.50)
    backwall = img[bw_y, bw_x]
    print(f"  Back wall ref RGB=({backwall[0]:.3f},{backwall[1]:.3f},{backwall[2]:.3f})")

    for label, mask, cx, cy in (("LEFT", left_mask, cx_l, cy_l), ("RIGHT", right_mask, cx_r, cy_r)):
        region_rgb = img[mask]
        dists = np.linalg.norm(region_rgb - backwall, axis=1)
        min_i = int(np.argmin(dists))
        coords = np.argwhere(mask)
        ty, tx = coords[min_i]
        silhouette = img[cy, cx]
        print(f"  {label} silhouette RGB=({silhouette[0]:.3f},{silhouette[1]:.3f},{silhouette[2]:.3f}) "
              f"L2={np.linalg.norm(silhouette-backwall):.3f}")
        print(f"  {label} best-transmit @ ({tx},{ty}) RGB=({region_rgb[min_i,0]:.3f},{region_rgb[min_i,1]:.3f},"
              f"{region_rgb[min_i,2]:.3f}) L2_to_backwall={dists[min_i]:.3f}")


def analyze_grazing(path, floor_y_frac, refl_x_frac, refl_y_frac, refl_r_frac=0.08):
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    print(f"\n=== {path} ({w}x{h}) [grazing floor] ===")

    y0 = int(h * floor_y_frac)
    y1 = min(h - 1, y0 + max(8, h // 40))
    floor_band = img[y0:y1, :]
    floor_lum = (0.2126 * floor_band[:, :, 0] + 0.7152 * floor_band[:, :, 1]
                 + 0.0722 * floor_band[:, :, 2]).mean()
    floor_r = floor_band[:, :, 0].mean()
    print(f"  Floor band y=[{y0},{y1}] lum={floor_lum:.3f} mean R={floor_r:.3f}")

    cx = int(w * refl_x_frac)
    cy = int(h * refl_y_frac)
    r = int(min(w, h) * refl_r_frac)
    ys = np.arange(h)
    xs = np.arange(w)
    xx, yy = np.meshgrid(xs, ys)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    region = img[mask]
    mean = region.mean(axis=0)
    lum = (0.2126 * region[:, 0] + 0.7152 * region[:, 1] + 0.0722 * region[:, 2]).mean()
    print(f"  Reflection ROI @ ({cx},{cy}) r={r}: RGB=({mean[0]:.3f},{mean[1]:.3f},{mean[2]:.3f}) lum={lum:.3f}")
    return floor_lum, lum, mean[0]


if __name__ == "__main__":
    grazing_lums = {}
    for p in sys.argv[1:]:
        name = p.split("/")[-1]
        if "grazing_topdown" in name:
            fl, rl, rr = analyze_grazing(p, floor_y_frac=0.52, refl_x_frac=0.42,
                                         refl_y_frac=0.50, refl_r_frac=0.05)
            grazing_lums["topdown"] = (fl, rl, rr)
        elif "grazing_low" in name:
            fl, rl, rr = analyze_grazing(p, floor_y_frac=0.58, refl_x_frac=0.50,
                                         refl_y_frac=0.585, refl_r_frac=0.04)
            grazing_lums["low"] = (fl, rl, rr)
        elif "debug_transmit" in name:
            analyze_single_sphere(p)
        else:
            analyze_spheres(p)

    if "topdown" in grazing_lums and "low" in grazing_lums:
        _, rl_top, rr_top = grazing_lums["topdown"]
        _, rl_low, rr_low = grazing_lums["low"]
        print(f"\n=== Grazing A vs B ===")
        print(f"  Floor reflection lum: topdown={rl_top:.3f}  low={rl_low:.3f}  ratio={rl_low/max(rl_top,1e-6):.2f}x")
        print(f"  Floor red channel (reflection ROI): topdown R={rr_top:.3f}  low R={rr_low:.3f}")
