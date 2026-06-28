#!/usr/bin/env python3
"""BMP→PNG, labels, and grazing composite for Fresnel Cornell renders."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

LABELS = {
    "fresnel_cornell_compare": ("noFresnel (L)", "Fresnel ON (R)"),
    "fresnel_cornell_water_glass": ("Water IOR 1.33 (L)", "Glass IOR 1.52 (R)"),
    "fresnel_grazing_topdown": ("Top-down glass floor", "see red ball below"),
    "fresnel_grazing_low": ("Grazing angle (F→1)", "mirror floor + red ball"),
}


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


def add_labels(bmp: Path, out: Path, left: str, right: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    w, h, px = read_bmp(bmp)
    img = pixels_to_image(px, w, h)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    pad = 16
    for text, cx_frac in ((left, 0.25), (right, 0.75)):
        cx = int(w * cx_frac)
        tw = draw.textlength(text, font=font)
        x = int(cx - tw / 2)
        y = pad
        draw.rectangle((x - 8, y - 4, x + tw + 8, y + 32), fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)


def add_single_label(bmp: Path, out: Path, label: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    w, h, px = read_bmp(bmp)
    img = pixels_to_image(px, w, h)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    pad = 16
    tw = draw.textlength(label, font=font)
    x = int(w / 2 - tw / 2)
    draw.rectangle((x - 8, pad - 4, x + tw + 8, pad + 32), fill=(0, 0, 0))
    draw.text((x, pad), label, fill=(255, 255, 255), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)


def grazing_composite(d: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    top = d / "fresnel_grazing_topdown.png"
    low = d / "fresnel_grazing_low.png"
    if not top.is_file() or not low.is_file():
        print(f"Missing grazing PNGs for composite", file=sys.stderr)
        return
    a = Image.open(top)
    b = Image.open(low)
    gap = 8
    out_w = a.width + b.width + gap
    out_h = max(a.height, b.height)
    canvas = Image.new("RGB", (out_w, out_h), (16, 16, 20))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    labels = (
        ("A: Top-down (transparent floor)", a.width // 2),
        ("B: Grazing (mirror floor)", a.width + gap + b.width // 2),
    )
    for text, cx in labels:
        tw = draw.textlength(text, font=font)
        x = int(cx - tw / 2)
        draw.rectangle((x - 6, 8, x + tw + 6, 40), fill=(0, 0, 0))
        draw.text((x, 12), text, fill=(255, 255, 255), font=font)
    out = d / "fresnel_grazing_compare.png"
    canvas.save(out, optimize=True)
    print(f"Wrote {out}")


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
        "fresnel_grazing_topdown",
        "fresnel_grazing_low",
    ]
    for stem in stems:
        bmp = d / f"{stem}.bmp"
        if not bmp.is_file():
            print(f"Missing: {bmp}", file=sys.stderr)
            return 1
        save_png_from_bmp(bmp, d / f"{stem}.png")
        print(f"Wrote {d / f'{stem}.png'}")
        if stem in LABELS:
            left, right = LABELS[stem]
            if stem.startswith("fresnel_grazing"):
                add_single_label(bmp, d / f"{stem}_labeled.png", f"{left} — {right}")
            else:
                add_labels(bmp, d / f"{stem}_labeled.png", left, right)
            print(f"Wrote {d / f'{stem}_labeled.png'}")

    grazing_composite(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
