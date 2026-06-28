PA1-2 Final outputs
====================

final_simple — scene_final_simple.txt (§2.15 main showcase)
classic_mis  — scene_classic_mis.txt (§2.15.4 classic 5-sphere Cornell)

Classic Cornell MIS
-------------------
Scene: testcases/scene_classic_mis.txt
Sphere notes: testcases/scene_classic_mis.README.txt

Changes from scene_path.txt:
  - Top-left blue sphere (-0.52, 1.12, -0.42) → dispersive glass IOR 1.50, delta 0.10
  - Bottom-right mirror sphere (0.60, 0.38, 0.40) → gold GGX glossy
  - Top-right mirror, red/green spheres, glass cube — unchanged

Render:
  ./build/PA1-2 testcases/scene_classic_mis.txt output/final/classic_mis.bmp \
      path_mis 128 gamma cuda dispersion
  # CPU fallback: path_mis 128 gamma omp dispersion

Outputs:
  output/final/classic_mis.bmp / classic_mis.png
  results/classic_mis.png
