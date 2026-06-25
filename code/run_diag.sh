#!/usr/bin/env bash
set -e
cd /data/PA1-2/code
mkdir -p output/diag
B=./build/PA1-2
CMP=python3 compare_bmp.py

run_pair() {
  local scene="$1" mode="$2" spp="$3"
  local extra="${4:-}"
  local tag="${scene}_${mode}_${spp}${extra// /_}"
  echo "=== $tag ==="
  $B testcases/${scene}.txt output/diag/${tag}_cpu.bmp $mode $spp $extra 2>&1 | grep "Render time"
  $B testcases/${scene}.txt output/diag/${tag}_gpu.bmp $mode $spp $extra cuda 2>&1 | grep "Render time"
  $CMP output/diag/${tag}_cpu.bmp output/diag/${tag}_gpu.bmp
  echo
}

run_pair scene01_basic whitted 1
run_pair scene_path path_nee 64
run_pair scene_glossy path_nee 64
run_pair scene_showcase path_nee 64 gamma
run_pair scene_showcase path_nee 64 "gamma dispersion"
