#!/usr/bin/env bash
# Fresnel Cornell bonus renders (CUDA path_nee, 256 SPP, gamma).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p output/fresnel build

echo "Building PA1-2..."
(cd build && cmake .. && make -j)

render() {
  local scene="$1"
  local out="$2"
  echo "=== $scene -> $out ==="
  ./build/PA1-2 "$scene" "$out" path_nee 256 gamma cuda
}

render testcases/scene_fresnel_cornell_compare.txt \
       output/fresnel/fresnel_cornell_compare.bmp
render testcases/scene_fresnel_cornell_water_glass.txt \
       output/fresnel/fresnel_cornell_water_glass.bmp
render testcases/scene_fresnel_cornell_grazing.txt \
       output/fresnel/fresnel_cornell_grazing.bmp

python3 -c "from PIL import Image; import os; [Image.open(f'output/fresnel/{n}.bmp').save(f'output/fresnel/{n}.png') for n in ['fresnel_cornell_compare','fresnel_cornell_water_glass','fresnel_cornell_grazing']]"

python3 scripts/analyze_fresnel_regions.py \
  output/fresnel/fresnel_cornell_compare.png \
  output/fresnel/fresnel_cornell_water_glass.png \
  output/fresnel/fresnel_cornell_grazing.png

echo "Done. Outputs in output/fresnel/"
