#!/usr/bin/env python3
"""Path guiding self-tests: white furnace energy conservation + A/B brightness/noise."""
import math
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "build" / "PA1-2"


def read_bmp(path):
    with open(path, "rb") as f:
        header = f.read(14)
        dib = f.read(40)
        width, height = struct.unpack("<ii", dib[4:12])
        bpp = struct.unpack("<H", dib[14:16])[0]
        abs_h = abs(height)
        row_bytes = ((bpp * width + 31) // 32) * 4
        pixels = []
        for _ in range(abs_h):
            row = f.read(row_bytes)
            for x in range(width):
                b, g, r = row[x * 3 : x * 3 + 3]
                pixels.append((r / 255.0, g / 255.0, b / 255.0))
        if height > 0:
            pixels = pixels[::-1]
        return width, abs_h, pixels


def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def stats(path):
    w, h, px = read_bmp(path)
    lums = [luminance(p) for p in px]
    mean = sum(lums) / len(lums)
    var = sum((x - mean) ** 2 for x in lums) / len(lums)
    return {"w": w, "h": h, "mean": mean, "std": math.sqrt(var)}


def render(scene, out, mode, spp, train_spp=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), str(scene), str(out), mode, str(spp), "gamma", "cuda"]
    if train_spp is not None:
        cmd.extend(["train_spp", str(train_spp)])
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def bmp_to_png(bmp_path):
    png_path = bmp_path.with_suffix(".png")
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed; skip PNG", bmp_path)
        return
    w, h, px = read_bmp(bmp_path)
    img = Image.new("RGB", (w, h))
    img.putdata([(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in px])
    img.save(png_path)
    print("Wrote", png_path)


def main():
    if not BIN.is_file():
        print("Build first: cmake -B build && cmake --build build", file=sys.stderr)
        return 1

    out = ROOT / "output"
    failures = []

    # 1) White furnace — sphere should be nearly invisible (uniform field).
    wf = out / "white_furnace_guided.bmp"
    render(ROOT / "testcases/scene_white_furnace.txt", wf, "path_guiding", 64, train_spp=128)
    wf_s = stats(wf)
    w, h = wf_s["w"], wf_s["h"]
    cx, cy = w // 2, h // 2
    r = min(w, h) // 8
    _, _, px = read_bmp(wf)
    center = [luminance(px[y * w + x]) for y in range(cy - r, cy + r) for x in range(cx - r, cx + r)]
    c_mean = sum(center) / len(center)
    c_std = math.sqrt(sum((x - c_mean) ** 2 for x in center) / len(center))
    print(f"White furnace: global mean={wf_s['mean']:.4f} std={wf_s['std']:.4f} "
          f"center mean={c_mean:.4f} center std={c_std:.4f}")
    if c_std > 0.12 or wf_s["mean"] < 0.1:
        failures.append(f"white furnace not uniform (mean={wf_s['mean']:.4f} center_std={c_std:.4f})")

    # 2) A/B Cornell — brightness within 5%, guiding lower or similar std.
    nee = out / "guiding_nee_128.bmp"
    guided = out / "guiding_guided_128.bmp"
    render(ROOT / "testcases/scene_path.txt", nee, "path_nee", 128)
    render(ROOT / "testcases/scene_path.txt", guided, "path_guiding", 128, train_spp=128)
    for p in (nee, guided):
        bmp_to_png(p)
    s_nee = stats(nee)
    s_gd = stats(guided)
    ratio = s_gd["mean"] / max(s_nee["mean"], 1e-6)
    print(f"A/B 128spp: nee mean={s_nee['mean']:.4f} std={s_nee['std']:.4f} | "
          f"guided mean={s_gd['mean']:.4f} std={s_gd['std']:.4f} | ratio={ratio:.3f}")
    if abs(ratio - 1.0) > 0.05:
        failures.append(f"brightness ratio {ratio:.3f} outside 5%")
    if s_gd["std"] > s_nee["std"] * 1.02:
        failures.append(f"guiding std {s_gd['std']:.4f} not lower than nee {s_nee['std']:.4f}")

    if failures:
        print("FAILED:", "; ".join(failures))
        return 1
    print("All guiding self-tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
