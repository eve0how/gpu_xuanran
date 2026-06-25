# #!/usr/bin/env bash

# cmake -B build
# cmake --build build

# # Run all testcases. 
# # You can comment some lines to disable the run of specific examples.
# mkdir -p output
# build/PA1-2 testcases/scene01_basic.txt output/scene01.bmp
# build/PA1-2 testcases/scene02_cube.txt output/scene02.bmp
# build/PA1-2 testcases/scene03_sphere.txt output/scene03.bmp
# build/PA1-2 testcases/scene04_axes.txt output/scene04.bmp
# build/PA1-2 testcases/scene05_bunny_200.txt output/scene05.bmp
# build/PA1-2 testcases/scene06_bunny_1k.txt output/scene06.bmp
# build/PA1-2 testcases/scene07_shine.txt output/scene07.bmp
# build/PA1-2 testcases/scene08_whitted.txt output/scene08_whitted.bmp whitted
# build/PA1-2 testcases/scene08_path.txt output/scene08_path.bmp path 64
# build/PA1-2 testcases/scene08_path.txt output/scene08_path_nee.bmp path_nee 64
# build/PA1-2 testcases/scene09_glossy.txt output/scene09_glossy.bmp path_nee 64

#!/usr/bin/env bash
set -e
cd /data/PA1-2/code
mkdir -p output
B=./build/PA1-2

# 1. 基础 Whitted
for s in scene01_basic scene02_cube scene03_sphere scene04_axes \
         scene05_bunny_200 scene06_bunny_1k scene07_shine; do
  $B testcases/${s}.txt output/${s}_cuda.bmp whitted cuda
done

# 2. Cornell Whitted
$B testcases/scene08_whitted.txt output/scene08_whitted_cuda.bmp whitted cuda
$B testcases/scene_whitted.txt output/scene_whitted_cuda.bmp whitted cuda

# 3. 路径追踪
$B testcases/scene08_path.txt output/scene08_path_cuda.bmp path 64 cuda
$B testcases/scene08_path.txt output/scene08_path_nee_cuda.bmp path_nee 64 cuda
$B testcases/scene_path.txt output/scene_path_no_nee_cuda.bmp path 64 cuda
$B testcases/scene_path.txt output/scene_path_nee_cuda.bmp path_nee 64 cuda

# 4. 光泽
$B testcases/scene09_glossy.txt output/scene09_glossy_cuda.bmp path_nee 64 cuda
$B testcases/scene_glossy.txt output/scene_glossy_cuda.bmp path_nee 64 cuda
$B testcases/scene_glossy.txt output/scene_glossy_mis_cuda.bmp path_mis 64 cuda

# 5. MIS 三策略
$B testcases/scene_mis_demo.txt output/mis_demo_brdf_32_cuda.bmp path 32 gamma cuda
$B testcases/scene_mis_demo.txt output/mis_demo_nee_32_cuda.bmp path_nee 32 gamma cuda
$B testcases/scene_mis_demo.txt output/mis_demo_mis_32_cuda.bmp path_mis 32 gamma cuda

# 6. 终局展示 + 色散
$B testcases/scene_showcase.txt output/showcase_path_nee_cuda.bmp path_nee 64 gamma cuda
$B testcases/scene_showcase.txt output/showcase_path_mis_cuda.bmp path_mis 64 gamma cuda
$B testcases/scene_showcase.txt output/dispersion_before_cuda.bmp path_nee 64 gamma cuda
$B testcases/scene_showcase.txt output/dispersion_after_cuda.bmp path_nee 64 gamma dispersion cuda

# 7. 纹理（需先 ./build/gen_textures）
./build/gen_textures 2>/dev/null || true
$B testcases/scene_texture.txt output/scene_texture_cuda.bmp whitted cuda
$B testcases/scene_texture_mesh_before.txt output/texture_mesh_before_cuda.bmp whitted cuda
$B testcases/scene_texture_mesh.txt output/texture_mesh_after_cuda.bmp whitted cuda
$B testcases/scene_texture_mesh_normal.txt output/normal_mesh_cuda.bmp whitted cuda

echo "All CUDA renders done. Check output/"