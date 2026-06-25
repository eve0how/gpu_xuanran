PA1-2 Submission Renders
========================

Scene tweaks (subtle, no major optical changes)
-----------------------------------------------
Applied consistently to scene_whitted.txt, scene_path.txt, scene_glossy.txt,
and synced scene08_whitted.txt / scene08_path.txt:

  Camera FOV: 36 -> 35.5 degrees

  Point light (Whitted / Glossy): color 2 2 2 -> 2.1 1.95 1.85 (slightly warm)

  Area light + emissive ceiling (Path): 80 80 80 -> 82 78 76 (slightly warm)

  Wall / floor materials:
    Floor (mat 0):         0.725 0.725 0.725 -> 0.70 0.73 0.68  (cooler cream)
    Left red wall (mat 2): 0.630 0.065 0.050 -> 0.58 0.08 0.06   (less saturated)
    Back wall (mat 8/9):   0.65 0.65 0.60 -> 0.63 0.64 0.58      (slightly warmer)
    Red sphere (mat 4):    0.800 0.150 0.150 -> 0.75 0.17 0.16
    Green wall (mat 5):    0.150 0.680 0.200 -> 0.12 0.60 0.22

  Sphere positions (approx +/-0.03) and minor radius adjustments:
    Red:     center 0.05 0.70 0.05  r=0.40 -> 0.08 0.68 0.03  r=0.38
    Blue:    center -0.52 1.12 -0.42 r=0.33 -> -0.49 1.10 -0.45 r=0.31
    Green:   center -0.55 0.42 0.58 r=0.28 -> -0.52 0.44 0.55 r=0.30
    Mirror:  center 0.62 1.15 -0.30 r=0.38 -> 0.59 1.13 -0.33 r=0.36
    Mirror2: center 0.60 0.38 0.40 r=0.26 -> 0.63 0.40 0.37 r=0.28
    Glass cube: Translate -0.55 0.36 0.62 scale 0.36 -> -0.52 0.36 0.59 scale 0.34
                (cube Y=0.36 kept ground-contact)

  Unchanged: IOR 1.45, mirror reflectivity, glossy roughness/F0, RR settings.

Render commands (from code/)
----------------------------
  build/PA1-2 testcases/scene_whitted.txt ../submit/whitted.bmp whitted
  build/PA1-2 testcases/scene_path.txt ../submit/path_no_nee.bmp path 64 gamma omp
  build/PA1-2 testcases/scene_path.txt ../submit/path_nee.bmp path_nee 64 gamma omp
  build/PA1-2 testcases/scene_glossy.txt ../submit/glossy.bmp path_nee 64 gamma omp

Output files
------------
  whitted.bmp      - Whitted-style ray tracing, scene_whitted.txt
  path_no_nee.bmp  - Path tracing without NEE, SPP=64, gamma, OpenMP
  path_nee.bmp     - Path tracing with NEE, SPP=64, gamma, OpenMP
  glossy.bmp       - Cook-Torrance glossy scene, path_nee SPP=64, gamma, OpenMP

MIS sampling comparison (scene_mis_demo.txt, SPP=32, gamma, OpenMP)
--------------------------------------------------------------------
  mis_compare_brdf_32.bmp - path: cos-weighted hemisphere + RR, NO NEE (BRDF sampling only)
  mis_compare_nee_32.bmp  - path_nee: next event estimation only (light PDF in denominator)
  mis_compare_mis_32.bmp  - path_mis: balance heuristic combining light + BRDF PDFs

Optional lower-SPP versions: mis_compare_*_16.bmp (SPP=16).

Scene: Cornell box + 5 glossy spheres (gold/silver metal m=0.05, plastic m=0.20),
       point light + small area lights on ceiling, dark background.
       Designed to show variance differences at low SPP; showcase scene NEE≈MIS is expected (unbiased).

All images are 1024x1024 BMP (~3.1 MB each).
