#!/usr/bin/env python3
"""Stitch Cornell texture demo panels into output/texture_showcase.png."""

from __future__ import annotations

import argparse
import os
import struct
import sys


def read_bmp(path: str) -> tuple[bytes, int, int]:
    with open(path, "rb") as f:
        header = f.read(54)
        if header[:2] != b"BM":
            raise ValueError(f"Not a BMP: {path}")
        width = struct.unpack_from("<i", header, 18)[0]
        height = struct.unpack_from("<i", header, 22)[0]
        offset = struct.unpack_from("<I", header, 10)[0]
        bpp = struct.unpack_from("<H", header, 28)[0]
        if bpp != 24:
            raise ValueError(f"Expected 24-bit BMP: {path}")
        f.seek(offset)
        row_bytes = ((width * 3 + 3) // 4) * 4
        rows = []
        for _ in range(abs(height)):
            row = f.read(row_bytes)
            rows.append(row[: width * 3])
        if height > 0:
            rows.reverse()
        return b"".join(rows), width, abs(height)


def write_png(path: str, rgb: bytes, width: int, height: int) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("Pillow required: pip install pillow")

    img = Image.frombytes("RGB", (width, height), rgb)
    img.save(path, format="PNG", optimize=True)


def load_panel_font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stitch_panels(panels: list[tuple[str, str]], out_path: str, thumb: int = 512) -> None:
    from PIL import Image, ImageDraw

    images = []
    for label, bmp_path in panels:
        if not os.path.isfile(bmp_path):
            raise FileNotFoundError(bmp_path)
        rgb, w, h = read_bmp(bmp_path)
        img = Image.frombytes("RGB", (w, h), rgb)
        if w != thumb or h != thumb:
            img = img.resize((thumb, thumb), Image.Resampling.LANCZOS)
        images.append((label, img))

    label_h = 36
    gap = 12
    pad = 16
    total_w = pad * 2 + thumb * len(images) + gap * (len(images) - 1)
    total_h = pad * 2 + label_h + thumb
    canvas = Image.new("RGB", (total_w, total_h), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = load_panel_font(18)

    x = pad
    y = pad + label_h
    for label, img in images:
        draw.text((x, pad), label, fill=(230, 230, 235), font=font)
        canvas.paste(img, (x, y))
        x += thumb + gap

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"Wrote {out_path} ({total_w}x{total_h})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-dir",
        default=os.path.join(os.path.dirname(__file__), ".."),
        help="Path to code/ directory",
    )
    parser.add_argument("--thumb", type=int, default=512)
    args = parser.parse_args()
    code_dir = os.path.abspath(args.code_dir)
    out_dir = os.path.join(code_dir, "output")

    panels = [
        ("A  Classic Cornell (solid)", os.path.join(out_dir, "texture_cornell_notex.bmp")),
        ("B  Plaster wall + marble sphere", os.path.join(out_dir, "texture_cornell.bmp")),
        ("C  Normal-mapped back wall", os.path.join(out_dir, "texture_cornell_normal.bmp")),
    ]
    out_path = os.path.join(out_dir, "texture_showcase.png")
    stitch_panels(panels, out_path, thumb=args.thumb)

    results_dir = os.path.join(code_dir, "results")
    if os.path.isdir(results_dir):
        import shutil

        dst = os.path.join(results_dir, "texture_showcase.png")
        shutil.copy2(out_path, dst)
        print(f"Copied to {dst}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
