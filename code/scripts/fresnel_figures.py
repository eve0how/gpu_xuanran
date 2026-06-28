#!/usr/bin/env python3
"""BMP→PNG and rim zoom crops for Fresnel Cornell renders."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# Right sphere upper-left silhouette rim (1024² grazing compare).
GRAZING_RIM_ROI = (520, 60, 900, 340)
ZOOM_CROP_SIZE = 160
ZOOM_SCALE = 3


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


def zoom_rim(bmp: Path, out: Path, roi: tuple[int, int, int, int]) -> None:
    from PIL import Image

    w, h, px = read_bmp(bmp)
    img = pixels_to_image(px, w, h)
    x0, y0, x1, y1 = roi
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = ZOOM_CROP_SIZE // 2
    box = (cx - half, cy - half, cx + half, cy + half)
    crop = img.crop(box).resize(
        (ZOOM_CROP_SIZE * ZOOM_SCALE, ZOOM_CROP_SIZE * ZOOM_SCALE),
        Image.Resampling.NEAREST,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "fresnel",
    )
    args = parser.parse_args()
    d = args.dir

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Install Pillow: pip install pillow", file=sys.stderr)
        return 1

    stems = [
        "fresnel_cornell_compare",
        "fresnel_cornell_water_glass",
        "fresnel_cornell_grazing",
    ]
    for stem in stems:
        bmp = d / f"{stem}.bmp"
        if not bmp.is_file():
            print(f"Missing: {bmp}", file=sys.stderr)
            return 1
        save_png_from_bmp(bmp, d / f"{stem}.png")
        print(f"Wrote {d / f'{stem}.png'}")

    grazing = d / "fresnel_cornell_grazing.bmp"
    zoom_rim(grazing, d / "fresnel_cornell_grazing_rim_zoom.png", GRAZING_RIM_ROI)
    print(f"Wrote {d / 'fresnel_cornell_grazing_rim_zoom.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
