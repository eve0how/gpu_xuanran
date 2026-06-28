#!/usr/bin/env bash
# Fresnel Cornell + Ward metal BRDF bonus renders (CUDA path_nee).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p output/fresnel output/ward output/glossy build

echo "Building PA1-2..."
(cd build && cmake .. && make -j)

render() {
  local scene="$1"
  local out="$2"
  local spp="${3:-256}"
  echo "=== $scene -> $out (${spp} SPP) ==="
  ./build/PA1-2 "$scene" "$out" path_nee "$spp" gamma cuda
}

render testcases/scene_fresnel_cornell_compare.txt \
       output/fresnel/fresnel_cornell_compare.bmp 256
render testcases/scene_fresnel_cornell_water_glass.txt \
       output/fresnel/fresnel_cornell_water_glass.bmp 256
render testcases/scene_fresnel_debug_transmit.txt \
       output/fresnel/fresnel_debug_transmit.bmp 128
render testcases/scene_fresnel_grazing_topdown.txt \
       output/fresnel/fresnel_grazing_topdown.bmp 256
render testcases/scene_fresnel_grazing_low.txt \
       output/fresnel/fresnel_grazing_low.bmp 256

render testcases/scene_brdf_metal_compare.txt \
       output/ward/brdf_metal_compare.bmp 512
cp output/ward/brdf_metal_compare.bmp output/glossy/brdf_metal_compare.bmp

python3 scripts/fresnel_figures.py

python3 scripts/analyze_fresnel_regions.py \
  output/fresnel/fresnel_cornell_compare.png \
  output/fresnel/fresnel_cornell_water_glass.png \
  output/fresnel/fresnel_grazing_topdown.png \
  output/fresnel/fresnel_grazing_low.png

echo "Done. Outputs in output/fresnel/, output/ward/, output/glossy/"
