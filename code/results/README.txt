PA1-2 Report Figure Manifest (results/)
=======================================
All paths relative to code/. REPORT.md embeds images from this folder only.
Original BMP renders remain under output/; PNG copies here are report figures.

Standard resolution
-------------------
  1024 x 1024  — default for Cornell / demo scenes (scene_path.txt, scene_mis_demo.txt, …)
  512 x 512    — scene_bvh_bunny.txt (BVH / CUDA bunny timing demos)
  640 x 640    — §2.14 AA zoom crops (4× upscale of fixed 160×160 ROI)
  Composite    — side-by-side / zoom / showcase panels (variable; see notes per image)

Markdown preview (REPORT.md)
----------------------------
  Two-column compare tables: width="400" height="400"
  Three-column MIS (§2.8): width="320" height="320" each
  Within each comparison group, pixel dimensions match; mismatched preview was
  previously caused by mixed width= (270/360/370) and flex max-width styles.

Convert BMP → PNG:
  python3 -c "from PIL import Image; Image.open('PATH.bmp').save('results/NAME.png')"

§2.1 Path Guiding (scene_guiding_occluder.txt, 1024², CUDA)
------------------------------------------------------------
compare_128_side_by_side.png
  python3 scripts/guiding_compare_figures.py
  (renders mis_128.bmp path_mis 128 + guiding_128.bmp path_guiding 128 train_spp 256, then stitches)

compare_512_side_by_side.png
  Same script at 512 SPP / train_spp 1024.

compare_128_zoom_4x.png
  Same script; 4× ROI crop of central shadow patch.

§2.2 Texture / Normal map (CPU Whitted 1 SPP, gamma, 1024²)
------------------------------------------------------------
texture_cornell_notex.png
  ./build/PA1-2 testcases/scene_texture_cornell_notex.txt output/texture_cornell_notex.bmp whitted 1 gamma

texture_cornell.png
  ./build/PA1-2 testcases/scene_texture_cornell.txt output/texture_cornell.bmp whitted 1 gamma

texture_cornell_normal.png
  ./build/PA1-2 testcases/scene_texture_cornell_normal.txt output/texture_cornell_normal.bmp whitted 1 gamma

texture_showcase.png  (1592×580 composite)
  python3 scripts/make_texture_showcase.py

§2.3 BVH (scene_bvh_bunny.txt, 512², path_nee 128, CUDA)
----------------------------------------------------------
bvh_bunny_on.png
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/bvh_compare/bunny_gpu_bvh_on_path128.bmp path_nee 128 gamma cuda

bvh_bunny_off.png
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/bvh_compare/bunny_gpu_bvh_off_path128.bmp path_nee 128 cuda no_bvh

§2.6 Whitted vs Path (scene_path.txt, 1024², CPU)
--------------------------------------------------
whitted_compare_whitted.png
  ./build/PA1-2 testcases/scene_path.txt output/report/whitted_path_compare/whitted.bmp whitted 1 gamma omp

whitted_compare_path.png
  ./build/PA1-2 testcases/scene_path.txt output/report/whitted_path_compare/path.bmp path 128 gamma omp

§2.7 NEE (scene_path.txt, 1024², CPU, both 64 SPP)
----------------------------------------------------
path_no_nee_64.png
  ./build/PA1-2 testcases/scene_path.txt output/report/path_no_nee_64.bmp path 64 gamma omp

path_nee_cornell.png
  ./build/PA1-2 testcases/scene_path.txt output/report/path_nee_64.bmp path_nee 64 gamma omp

§2.8 MIS three-way (scene_mis_demo.txt, 1024², 32 SPP, CUDA)
--------------------------------------------------------------
mis_compare_brdf_32.png  — path (BRDF only, no NEE)
  ./build/PA1-2 testcases/scene_mis_demo.txt output/acceptance/mis_compare_brdf_32.bmp path 32 gamma cuda

mis_compare_nee_32.png  — path_nee (NEE, no MIS weights on indirect emissive)
  ./build/PA1-2 testcases/scene_mis_demo.txt output/acceptance/mis_compare_nee_32.bmp path_nee 32 gamma cuda

mis_compare_mis_32.png  — path_mis (NEE + power-heuristic MIS)
  ./build/PA1-2 testcases/scene_mis_demo.txt output/acceptance/mis_compare_mis_32.bmp path_mis 32 gamma cuda

§2.9 GGX Glossy (scene_glossy.txt, 1024²)
-------------------------------------------
glossy.png
  ./build/PA1-2 testcases/scene_glossy.txt output/glossy/glossy.bmp path_nee 64 gamma cuda

§2.10 Dispersion (scene_dispersion.txt, 1024²)
-----------------------------------------------
dispersion_before.png
  ./build/PA1-2 testcases/scene_dispersion.txt output/diag/dispersion_before.bmp path_nee 1024 gamma cuda

dispersion_after.png
  ./build/PA1-2 testcases/scene_dispersion.txt output/diag/dispersion_after.bmp path_nee 1024 gamma dispersion cuda

§2.11 Gamma (scene_path.txt path_nee 512, 1024², CPU)
------------------------------------------------------
path_nee_nogamma512.png
  ./build/PA1-2 testcases/scene_path.txt output/report/path_nee_nogamma512.bmp path_nee 512 omp

path_nee_512.png
  ./build/PA1-2 testcases/scene_path.txt output/report/path_nee_512.bmp path_nee 512 gamma omp

§2.12 OpenMP (1024², CPU)
--------------------------
accel_cpu_no_omp_path32.png  — scene_path.txt, path_nee 32 gamma
accel_cpu_omp_path32.png     — scene_path.txt, path_nee 32 gamma omp
accel_cpu_no_omp_whitted.png — scene_whitted.txt, whitted 1 gamma
accel_cpu_omp_whitted.png    — scene_whitted.txt, whitted 1 gamma omp

§2.13 CUDA vs CPU (scene_bvh_bunny.txt, 512²)
-----------------------------------------------
accel_cpu_bunny_whitted.png  — whitted 1 gamma omp
accel_gpu_bunny_whitted.png  — whitted 1 gamma cuda
accel_cpu_bunny_path64.png   — path_nee 64 gamma omp
accel_gpu_bunny_path64.png   — path_nee 64 gamma cuda

§2.14 Anti-aliasing (scene_whitted.txt, 1024², CUDA)
-----------------------------------------------------
aa_before.png / aa_after.png
  ./build/PA1-2 testcases/scene_whitted.txt output/report/aa_before.bmp whitted 1 cuda
  ./build/PA1-2 testcases/scene_whitted.txt output/report/aa_after.bmp whitted 16 cuda

aa_before_zoom.png / aa_after_zoom.png  (640×640 each)
  4× nearest upscale of ROI crop (400,320)–(560,480) from aa_before/aa_after.

§2.15 Final showcases (1024²)
------------------------------
final_simple.png  — scene_final_simple.txt, path_mis 64 gamma omp dispersion (CPU)
  ./build/PA1-2 testcases/scene_final_simple.txt output/final/final_simple.bmp path_mis 64 gamma omp dispersion

classic_mis.png  — scene_classic_mis.txt, path_mis 128 gamma cuda dispersion
  ./build/PA1-2 testcases/scene_classic_mis.txt output/final/classic_mis.bmp path_mis 128 gamma cuda dispersion

Appendix A — Fresnel (1024² unless noted)
------------------------------------------
fresnel_cornell_compare_labeled.png   — bash scripts/fresnel_render.sh + scripts/fresnel_figures.py
fresnel_cornell_water_glass_labeled.png
fresnel_grazing_compare.png           — 2056×1024 side-by-side (scripts/fresnel_figures.py)

Appendix B — Ward (1024²)
---------------------------
ward_brdf_metal_compare.png
  ./build/PA1-2 testcases/scene_brdf_metal_compare.txt output/ward/brdf_metal_compare.bmp path_nee 512 gamma cuda
