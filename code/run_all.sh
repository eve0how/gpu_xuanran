#!/usr/bin/env bash
# Run all testcases on CPU and GPU (CUDA).
# Build first: cmake -B build && cmake --build build -j$(nproc)
set -e
cd /data/PA1-2/code
mkdir -p output
B=./build/PA1-2

# echo "=== CPU renders ==="

# # 1. Basic Whitted
# for s in scene01_basic scene02_cube scene03_sphere scene04_axes \
#          scene05_bunny_200 scene06_bunny_1k scene07_shine; do
#   $B testcases/${s}.txt output/${s}.bmp whitted
# done

# # 2. Cornell Whitted
# $B testcases/scene08_whitted.txt output/scene08_whitted.bmp whitted
# $B testcases/scene_whitted.txt output/scene_whitted.bmp whitted

# # 3. Path tracing
# $B testcases/scene08_path.txt output/scene08_path.bmp path 64
# $B testcases/scene08_path.txt output/scene08_path_nee.bmp path_nee 64
# $B testcases/scene_path.txt output/scene_path_no_nee.bmp path 64
# $B testcases/scene_path.txt output/scene_path_nee.bmp path_nee 64

# # 4. Glossy
# $B testcases/scene09_glossy.txt output/scene09_glossy.bmp path_nee 64
# $B testcases/scene_glossy.txt output/scene_glossy.bmp path_nee 64
# $B testcases/scene_glossy.txt output/scene_glossy_mis.bmp path_mis 64

# # 5. MIS demo
# $B testcases/scene_mis_demo.txt output/mis_demo_brdf_32.bmp path 32 gamma
# $B testcases/scene_mis_demo.txt output/mis_demo_nee_32.bmp path_nee 32 gamma
# $B testcases/scene_mis_demo.txt output/mis_demo_mis_32.bmp path_mis 32 gamma

# # 6. Showcase + dispersion
# $B testcases/scene_showcase.txt output/showcase_path_nee.bmp path_nee 64 gamma
# $B testcases/scene_showcase.txt output/showcase_path_mis.bmp path_mis 64 gamma
# $B testcases/scene_showcase.txt output/dispersion_before.bmp path_nee 64 gamma
# $B testcases/scene_showcase.txt output/dispersion_after.bmp path_nee 64 gamma dispersion

# # 7. Textures (generate assets first)
# ./build/gen_textures 2>/dev/null || true
# $B testcases/scene_texture.txt output/scene_texture.bmp whitted
# $B testcases/scene_texture_mesh_before.txt output/texture_mesh_before.bmp whitted
# $B testcases/scene_texture_mesh.txt output/texture_mesh_after.bmp whitted
# $B testcases/scene_texture_mesh_normal.txt output/normal_mesh.bmp whitted

echo "=== CUDA renders ==="

# 1. Basic Whitted
for s in scene01_basic scene02_cube scene03_sphere scene04_axes \
         scene05_bunny_200 scene06_bunny_1k scene07_shine; do
  $B testcases/${s}.txt output/${s}_cuda.bmp whitted cuda
done

# 2. Cornell Whitted
$B testcases/scene08_whitted.txt output/scene08_whitted_cuda.bmp whitted cuda
$B testcases/scene_whitted.txt output/scene_whitted_cuda.bmp whitted cuda

# 3. Path tracing
$B testcases/scene08_path.txt output/scene08_path_cuda.bmp path 64 cuda
$B testcases/scene08_path.txt output/scene08_path_nee_cuda.bmp path_nee 64 cuda
$B testcases/scene_path.txt output/scene_path_no_nee_cuda.bmp path 64 cuda
$B testcases/scene_path.txt output/scene_path_nee_cuda.bmp path_nee 64 cuda

# 4. Glossy
$B testcases/scene09_glossy.txt output/scene09_glossy_cuda.bmp path_nee 64 cuda
$B testcases/scene_glossy.txt output/scene_glossy_cuda.bmp path_nee 64 cuda
$B testcases/scene_glossy.txt output/scene_glossy_mis_cuda.bmp path_mis 64 cuda

# 5. MIS demo
$B testcases/scene_mis_demo.txt output/mis_demo_brdf_32_cuda.bmp path 32 gamma cuda
$B testcases/scene_mis_demo.txt output/mis_demo_nee_32_cuda.bmp path_nee 32 gamma cuda
$B testcases/scene_mis_demo.txt output/mis_demo_mis_32_cuda.bmp path_mis 32 gamma cuda

# 6. Showcase + dispersion
$B testcases/scene_showcase.txt output/showcase_path_nee_cuda.bmp path_nee 64 gamma cuda
$B testcases/scene_showcase.txt output/showcase_path_mis_cuda.bmp path_mis 64 gamma cuda
$B testcases/scene_showcase.txt output/dispersion_before_cuda.bmp path_nee 64 gamma cuda
$B testcases/scene_showcase.txt output/dispersion_after_cuda.bmp path_nee 64 gamma dispersion cuda

# 7. Textures
./build/gen_textures 2>/dev/null || true
$B testcases/scene_texture.txt output/scene_texture_cuda.bmp whitted cuda
$B testcases/scene_texture_mesh_before.txt output/texture_mesh_before_cuda.bmp whitted cuda
$B testcases/scene_texture_mesh.txt output/texture_mesh_after_cuda.bmp whitted cuda
$B testcases/scene_texture_mesh_normal.txt output/normal_mesh_cuda.bmp whitted cuda

echo "All CPU and CUDA renders done. Check output/"
