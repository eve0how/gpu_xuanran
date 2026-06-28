#!/usr/bin/env python3
"""Equal-SPP path_mis vs path_guiding figures for scene_guiding_occluder."""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

# Central occluded floor patch (1024² renders, camera aligned with Cornell demo).
SHADOW_ROI = (416, 563, 608, 737)  # x0, y0, x1, y1
ZOOM_CROP_SIZE = 128
ZOOM_SCALE = 4


def read_bmp(path: Path) -> tuple[int, int, list[tuple[float, float, float]]]:
    with open(path, "rb") as f:
        if f.read(2) != b"BM":
            raise ValueError(f"Not BMP: {path}")
        f.seek(10)
        offset = struct.unpack("<I", f.read(4))[0]
        f.seek(18)
        width, height = struct.unpack("<ii", f.read(8))
        f.seek(28)
        bpp = struct.unpack("<H", f.read(2))[0]
        if bpp != 24:
            raise ValueError(f"Expected 24-bit BMP: {path}")
        abs_h = abs(height)
        f.seek(offset)
        row_bytes = ((width * 3 + 3) // 4) * 4
        pixels: list[tuple[float, float, float]] = []
        for _ in range(abs_h):
            row = f.read(row_bytes)
            for x in range(width):
                b, g, r = row[x * 3 : x * 3 + 3]
                pixels.append((r / 255.0, g / 255.0, b / 255.0))
        if height > 0:
            pixels = pixels[::-1]
        return width, abs_h, pixels


def luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def roi_metrics(px: list, w: int, h: int, roi: tuple[int, int, int, int]) -> dict:
    x0, y0, x1, y1 = roi
    lums = [luminance(px[y * w + x]) for y in range(y0, y1) for x in range(x0, x1)]
    mean = sum(lums) / len(lums)
    std = math.sqrt(sum((x - mean) ** 2 for x in lums) / len(lums))
    return {"mean": mean, "std": std, "n": len(lums)}


def pixels_to_image(px, w, h):
    from PIL import Image

    img = Image.new("RGB", (w, h))
    img.putdata([(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in px])
    return img


def save_png_from_bmp(bmp: Path, png: Path) -> None:
    w, h, px = read_bmp(bmp)
    img = pixels_to_image(px, w, h)
    png.parent.mkdir(parents=True, exist_ok=True)
    img.save(png, optimize=True)


def side_by_side(left: Path, right: Path, out: Path, labels: tuple[str, str]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    w, h, px_l = read_bmp(left)
    _, _, px_r = read_bmp(right)
    img_l = pixels_to_image(px_l, w, h)
    img_r = pixels_to_image(px_r, w, h)
    label_h = 40
    gap = 8
    canvas = Image.new("RGB", (w * 2 + gap, h + label_h), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    canvas.paste(img_l, (0, label_h))
    canvas.paste(img_r, (w + gap, label_h))
    draw.text((12, 8), labels[0], fill=(240, 240, 245), font=font)
    draw.text((w + gap + 12, 8), labels[1], fill=(240, 240, 245), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)


def zoom_pair(left: Path, right: Path, out: Path, roi: tuple[int, int, int, int]) -> None:
    from PIL import Image

    w, h, _ = read_bmp(left)
    x0, y0, x1, y1 = roi
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = ZOOM_CROP_SIZE // 2
    box = (cx - half, cy - half, cx + half, cy + half)

    def crop(path: Path):
        _, _, px = read_bmp(path)
        img = pixels_to_image(px, w, h)
        return img.crop(box).resize(
            (ZOOM_CROP_SIZE * ZOOM_SCALE, ZOOM_CROP_SIZE * ZOOM_SCALE),
            Image.Resampling.NEAREST,
        )

    im_l = crop(left)
    im_r = crop(right)
    gap = 6
    canvas = Image.new("RGB", (im_l.width * 2 + gap, im_l.height), (20, 20, 24))
    canvas.paste(im_l, (0, 0))
    canvas.paste(im_r, (im_l.width + gap, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "guiding_compare",
    )
    args = parser.parse_args()
    d = args.dir

    pairs = [
        ("128", d / "mis_128.bmp", d / "guiding_128.bmp"),
        ("512", d / "mis_512.bmp", d / "guiding_512.bmp"),
    ]
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Install Pillow: pip install pillow", file=sys.stderr)
        return 1

    print("Shadow ROI (pixels):", SHADOW_ROI)
    for tag, mis, guided in pairs:
        if not mis.is_file() or not guided.is_file():
            print(f"Missing renders for {tag} spp", file=sys.stderr)
            return 1
        save_png_from_bmp(mis, mis.with_suffix(".png"))
        save_png_from_bmp(guided, guided.with_suffix(".png"))
        side_by_side(
            mis,
            guided,
            d / f"compare_{tag}_side_by_side.png",
            (f"path_mis ({tag} SPP)", f"path_guiding ({tag} SPP)"),
        )
        zoom_pair(mis, guided, d / f"compare_{tag}_zoom_4x.png", SHADOW_ROI)

        w, h, px_m = read_bmp(mis)
        _, _, px_g = read_bmp(guided)
        sm = roi_metrics(px_m, w, h, SHADOW_ROI)
        sg = roi_metrics(px_g, w, h, SHADOW_ROI)
        ratio = sg["mean"] / max(sm["mean"], 1e-9)
        std_drop = (1.0 - sg["std"] / max(sm["std"], 1e-9)) * 100.0
        print(
            f"[{tag} SPP] full mis mean={roi_metrics(px_m,w,h,(0,0,w,h))['mean']:.4f} "
            f"guided={roi_metrics(px_g,w,h,(0,0,w,h))['mean']:.4f}"
        )
        print(
            f"  shadow ROI: mis mean={sm['mean']:.4f} std={sm['std']:.4f} | "
            f"guided mean={sg['mean']:.4f} std={sg['std']:.4f} | "
            f"brightness ratio={ratio:.3f} | std reduction={std_drop:.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
