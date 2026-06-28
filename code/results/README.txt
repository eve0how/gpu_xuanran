PA1-2 Report Figure Manifest (results/)
=======================================
All paths relative to code/. REPORT.md embeds images from this folder only.
Original BMP renders remain under output/; PNG copies here are report figures.

Standard resolution
-------------------
  1024 x 1024  — default for Cornell / demo scenes (scene_path.txt, scene_mis_demo.txt, …)
  512 x 512    — scene_bvh_bunny.txt (BVH / CUDA bunny timing demos)
  640 x 640    — §2.14 AA zoom crops (4× upscale of fixed 160×160 ROI)
  Composite    — side-by-side / zoom panels (variable; see notes per image)

Convert BMP → PNG:
  python3 -c "from PIL import Image; Image.open('PATH.bmp').save('results/NAME.png')"

Full appendix: see REPORT.md §附录 (run from code/ after cmake --build build).
Reproduction policy: use **cuda** everywhere except §2.12 OpenMP timing, §2.13 CPU vs GPU timing, §2.2 textures (CPU only).

§2.1 Path Guiding (scene_guiding_occluder.txt, 1024², CUDA)
------------------------------------------------------------
compare_128_side_by_side.png
compare_512_side_by_side.png
compare_128_zoom_4x.png
  mkdir -p output/guiding_compare results
  ./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/mis_128.bmp path_mis 128 gamma cuda
  ./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/guiding_128.bmp path_guiding 128 gamma cuda train_spp 256
  ./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/mis_512.bmp path_mis 512 gamma cuda
  ./build/PA1-2 testcases/scene_guiding_occluder.txt output/guiding_compare/guiding_512.bmp path_guiding 512 gamma cuda train_spp 1024
  python3 scripts/guiding_compare_figures.py
  cp output/guiding_compare/compare_128_side_by_side.png output/guiding_compare/compare_512_side_by_side.png output/guiding_compare/compare_128_zoom_4x.png results/

§2.2 Texture / Normal map (CPU Whitted 1 SPP, gamma, 1024²)
------------------------------------------------------------
texture_cornell_notex.png
texture_cornell.png
texture_cornell_normal.png
  ./build/gen_textures
  mkdir -p output results
  ./build/PA1-2 testcases/scene_texture_cornell_notex.txt output/texture_cornell_notex.bmp whitted 1 gamma
  ./build/PA1-2 testcases/scene_texture_cornell.txt output/texture_cornell.bmp whitted 1 gamma
  ./build/PA1-2 testcases/scene_texture_cornell_normal.txt output/texture_cornell_normal.bmp whitted 1 gamma
  python3 -c "from PIL import Image; Image.open('output/texture_cornell_notex.bmp').save('results/texture_cornell_notex.png'); Image.open('output/texture_cornell.bmp').save('results/texture_cornell.png'); Image.open('output/texture_cornell_normal.bmp').save('results/texture_cornell_normal.png')"

§2.3 BVH (scene_bvh_bunny.txt, 512², path_nee 128, CUDA)
----------------------------------------------------------
bvh_bunny_on.png
bvh_bunny_off.png
  mkdir -p output/bvh_compare results
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/bvh_compare/bunny_bvh_on.bmp path_nee 128 gamma cuda
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/bvh_compare/bunny_bvh_off.bmp path_nee 128 gamma cuda no_bvh
  python3 -c "from PIL import Image; Image.open('output/bvh_compare/bunny_bvh_on.bmp').save('results/bvh_bunny_on.png'); Image.open('output/bvh_compare/bunny_bvh_off.bmp').save('results/bvh_bunny_off.png')"

§2.6 Whitted vs path_nee (scene_whitted_path_compare.txt, 1024², CUDA)
----------------------------------------------------------------------
whitted_compare_whitted.png
whitted_compare_path.png
  mkdir -p output/report/whitted_path_compare results
  ./build/PA1-2 testcases/scene_whitted_path_compare.txt output/report/whitted_path_compare/whitted.bmp whitted 1 gamma cuda
  ./build/PA1-2 testcases/scene_whitted_path_compare.txt output/report/whitted_path_compare/path_nee.bmp path_nee 128 gamma cuda
  python3 -c "from PIL import Image; Image.open('output/report/whitted_path_compare/whitted.bmp').save('results/whitted_compare_whitted.png'); Image.open('output/report/whitted_path_compare/path_nee.bmp').save('results/whitted_compare_path.png')"

§2.7 NEE (scene_path.txt, 1024², CUDA, 256 SPP)
------------------------------------------------
path_256.png
path_nee_256.png
  mkdir -p output/report results
  ./build/PA1-2 testcases/scene_path.txt output/report/path_256.bmp path 256 gamma cuda
  ./build/PA1-2 testcases/scene_path.txt output/report/path_nee_256.bmp path_nee 256 gamma cuda
  python3 -c "from PIL import Image; Image.open('output/report/path_256.bmp').save('results/path_256.png'); Image.open('output/report/path_nee_256.bmp').save('results/path_nee_256.png')"

§2.8 MIS three-way (scene_mis_demo.txt, 1024², 32 SPP, CUDA)
--------------------------------------------------------------
mis_compare_brdf_32.png  — path (BRDF only)
mis_compare_nee_32.png   — path_nee
mis_compare_mis_32.png   — path_mis
  mkdir -p output/report/mis_compare results
  ./build/PA1-2 testcases/scene_mis_demo.txt output/report/mis_compare/brdf_32.bmp path 32 gamma cuda
  ./build/PA1-2 testcases/scene_mis_demo.txt output/report/mis_compare/nee_32.bmp path_nee 32 gamma cuda
  ./build/PA1-2 testcases/scene_mis_demo.txt output/report/mis_compare/mis_32.bmp path_mis 32 gamma cuda
  python3 -c "from PIL import Image; Image.open('output/report/mis_compare/brdf_32.bmp').save('results/mis_compare_brdf_32.png'); Image.open('output/report/mis_compare/nee_32.bmp').save('results/mis_compare_nee_32.png'); Image.open('output/report/mis_compare/mis_32.bmp').save('results/mis_compare_mis_32.png')"

§2.9 GGX Glossy (scene_glossy.txt, 1024², CUDA)
-----------------------------------------------
glossy.png
  mkdir -p output/glossy results
  ./build/PA1-2 testcases/scene_glossy.txt output/glossy/glossy.bmp path_nee 64 gamma cuda
  python3 -c "from PIL import Image; Image.open('output/glossy/glossy.bmp').save('results/glossy.png')"

§2.10 Dispersion (scene_dispersion.txt, 1024², CUDA)
----------------------------------------------------
dispersion_before.png / dispersion_after.png
  mkdir -p output/diag results
  ./build/PA1-2 testcases/scene_dispersion.txt output/diag/dispersion_before.bmp path_nee 128 gamma cuda
  ./build/PA1-2 testcases/scene_dispersion.txt output/diag/dispersion_after.bmp path_nee 128 gamma dispersion cuda
  python3 -c "from PIL import Image; Image.open('output/diag/dispersion_before.bmp').save('results/dispersion_before.png'); Image.open('output/diag/dispersion_after.bmp').save('results/dispersion_after.png')"

§2.11 Gamma (scene_path.txt path_nee 512, 1024², CUDA)
-------------------------------------------------------
path_nee_nogamma512.png
  ./build/PA1-2 testcases/scene_path.txt output/report/path_nee_nogamma512.bmp path_nee 512 cuda

path_nee_512.png
  ./build/PA1-2 testcases/scene_path.txt output/report/path_nee_512.bmp path_nee 512 gamma cuda

  python3 -c "from PIL import Image; Image.open('output/report/path_nee_nogamma512.bmp').save('results/path_nee_nogamma512.png'); Image.open('output/report/path_nee_512.bmp').save('results/path_nee_512.png')"

§2.12 OpenMP (1024², CPU)
--------------------------
accel_cpu_no_omp_path32.png  — scene_path.txt, path_nee 32 gamma
accel_cpu_omp_path32.png     — scene_path.txt, path_nee 32 gamma omp
accel_cpu_no_omp_whitted.png — scene_whitted.txt, whitted 1 gamma
accel_cpu_omp_whitted.png    — scene_whitted.txt, whitted 1 gamma omp
  mkdir -p output/report results
  ./build/PA1-2 testcases/scene_path.txt output/report/cpu_no_omp_path32.bmp path_nee 32 gamma
  ./build/PA1-2 testcases/scene_path.txt output/report/cpu_omp_path32.bmp path_nee 32 gamma omp
  ./build/PA1-2 testcases/scene_whitted.txt output/report/cpu_no_omp_whitted.bmp whitted 1 gamma
  ./build/PA1-2 testcases/scene_whitted.txt output/report/cpu_omp_whitted.bmp whitted 1 gamma omp
  python3 -c "from PIL import Image; Image.open('output/report/cpu_no_omp_path32.bmp').save('results/accel_cpu_no_omp_path32.png'); Image.open('output/report/cpu_omp_path32.bmp').save('results/accel_cpu_omp_path32.png'); Image.open('output/report/cpu_no_omp_whitted.bmp').save('results/accel_cpu_no_omp_whitted.png'); Image.open('output/report/cpu_omp_whitted.bmp').save('results/accel_cpu_omp_whitted.png')"

§2.13 CUDA vs CPU (scene_bvh_bunny.txt, 512²)
-----------------------------------------------
accel_cpu_bunny_whitted.png  — whitted 1 gamma omp
accel_gpu_bunny_whitted.png  — whitted 1 gamma cuda
accel_cpu_bunny_path64.png   — path_nee 64 gamma omp
accel_gpu_bunny_path64.png   — path_nee 64 gamma cuda
  mkdir -p output/report results
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/report/cpu_bunny_whitted.bmp whitted 1 gamma omp
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/report/gpu_bunny_whitted.bmp whitted 1 gamma cuda
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/report/cpu_bunny_path64.bmp path_nee 64 gamma omp
  ./build/PA1-2 testcases/scene_bvh_bunny.txt output/report/gpu_bunny_path64.bmp path_nee 64 gamma cuda
  python3 -c "from PIL import Image; Image.open('output/report/cpu_bunny_whitted.bmp').save('results/accel_cpu_bunny_whitted.png'); Image.open('output/report/gpu_bunny_whitted.bmp').save('results/accel_gpu_bunny_whitted.png'); Image.open('output/report/cpu_bunny_path64.bmp').save('results/accel_cpu_bunny_path64.png'); Image.open('output/report/gpu_bunny_path64.bmp').save('results/accel_gpu_bunny_path64.png')"

§2.14 Anti-aliasing (scene_whitted.txt, 1024², CUDA)
-----------------------------------------------------
aa_before.png / aa_after.png
  ./build/PA1-2 testcases/scene_whitted.txt output/report/aa_before.bmp whitted 1 cuda
  ./build/PA1-2 testcases/scene_whitted.txt output/report/aa_after.bmp whitted 16 cuda

aa_before_zoom.png / aa_after_zoom.png
  4× nearest upscale of ROI crop (400,320)–(560,480) from aa_before/aa_after:
  python3 -c "
  from PIL import Image
  for name in ('aa_before', 'aa_after'):
      img = Image.open(f'output/report/{name}.bmp')
      img.save(f'results/{name}.png')
      crop = img.crop((400, 320, 560, 480))
      crop.resize((crop.width * 4, crop.height * 4), Image.NEAREST).save(f'results/{name}_zoom.png')
  "

§2.15 Final showcases (1024²)
------------------------------
classic_mis.png   — scene_classic_mis.txt, path_mis 15000 gamma cuda dispersion
  mkdir -p output/final results
  ./build/PA1-2 testcases/scene_classic_mis.txt output/final/classic_mis.bmp path_mis 15000 gamma cuda dispersion
  python3 -c "from PIL import Image; Image.open('output/final/classic_mis.bmp').save('results/classic_mis.png')"
