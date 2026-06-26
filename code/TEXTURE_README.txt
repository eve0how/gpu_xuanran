PA1-2 Texture / Normal Map Bonus — Cornell Showcase
===================================================

One-file submission
-------------------
  output/texture_showcase.png

  Three panels stitched left-to-right:
    A  Classic Cornell (solid colors)     — scene_texture_cornell_notex.txt
    B  Plaster back wall + marble sphere  — scene_texture_cornell.txt
    C  Normal-mapped plaster back wall    — scene_texture_cornell_normal.txt

Scene layout (classic Cornell box, based on scene_whitted.txt)
--------------------------------------------------------------
  Floor / ceiling : solid gray / dark (planes)
  Left wall (x=-1) : classic red
  Right wall (x=+1) : classic green
  Back wall (z=-1) : textured plaster (TriangleMesh cornell_back_wall.obj)
  Center sphere   : marble albedo (sphere UV mapping)
  Other spheres   : blue / green / mirrors; glass cube unchanged

Generated textures (run gen_textures first)
-------------------------------------------
  textures/plaster_albedo.bmp   — soft off-white wall plaster (panel B albedo only)
  textures/plaster_normal.bmp — stronger bump normal map for panel C
  textures/marble_albedo.bmp    — blue-gray veined marble for center sphere
  textures/brick_*.bmp, wood_*.bmp, stone_*.bmp, earth_albedo.bmp — legacy demos

Commands (CPU path, from code/)
-------------------------------
  mkdir -p build output textures
  cd build && cmake .. && make gen_textures PA1-2 && cd ..
  ./build/gen_textures
  ./build/PA1-2 testcases/scene_texture_cornell_notex.txt output/texture_cornell_notex.bmp whitted 1 gamma
  ./build/PA1-2 testcases/scene_texture_cornell.txt output/texture_cornell.bmp whitted 1 gamma
  ./build/PA1-2 testcases/scene_texture_cornell_normal.txt output/texture_cornell_normal.bmp whitted 1 gamma
  python3 -c "from PIL import Image; import shutil; import os; os.makedirs('results',exist_ok=True); \
    [Image.open(f'output/{n}.bmp').save(f'output/{n}.png') or shutil.copy(f'output/{n}.png',f'results/{n}.png') \
     for n in ('texture_cornell_notex','texture_cornell','texture_cornell_normal')]"
  python3 scripts/make_texture_showcase.py

Individual renders (required submission files)
----------------------------------------------
  output/texture_cornell_notex.png   — Panel A (solid Cornell)
  output/texture_cornell.png         — Panel B (plaster albedo back wall)
  output/texture_cornell_normal.png  — Panel C (plaster + normal map)

  results/texture_cornell_notex.png
  results/texture_cornell.png
  results/texture_cornell_normal.png

Notes
-----
  Textures are sampled on CPU only (Whitted / path / path_nee). Do not pass cuda/gpu.
  Normal maps require mesh UV + TBN; back wall uses cornell_back_wall.obj.
  Panel B (scene_texture_cornell.txt): back wall material has texture only, no normalMap.
  Panel C (scene_texture_cornell_normal.txt): same albedo + normalMap, lower diffuse and
  slight specular so bump shading is visible under the Whitted point light.
