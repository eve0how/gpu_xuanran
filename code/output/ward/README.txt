Ward Anisotropic BRDF Bonus Renders (256 SPP, path_nee, CUDA, gamma)
===================================================================

Build:
  cd /data/PA1-2/code/build && cmake .. && make -j

Recommended batch (includes Fresnel + Ward):
  cd /data/PA1-2/code && bash scripts/fresnel_render.sh

Assignment pair (2 images — tangent rotation):
  ./build/PA1-2 testcases/scene_ward_aniso_A.txt \
      output/ward/ward_aniso_A.bmp path_nee 256 gamma cuda

  ./build/PA1-2 testcases/scene_ward_aniso_B.txt \
      output/ward/ward_aniso_B.bmp path_nee 256 gamma cuda

Optional 3-panel comparison (clearest visual proof):
  ./build/PA1-2 testcases/scene_ward_aniso_showcase.txt \
      output/ward/ward_aniso_showcase.bmp path_nee 256 gamma cuda

Analysis:
  python3 scripts/ward_ab_analysis.py output/ward/ward_aniso_A.bmp output/ward/ward_aniso_B.bmp
  python3 scripts/ward_showcase_analysis.py output/ward/ward_aniso_showcase.bmp

----------------------------------------------------------------------
1. ward_aniso_A.bmp
   Scene: testcases/scene_ward_aniso_A.txt
   Gold metal plane (kd=0, ks gold), alphaX=0.04, alphaY=0.45, tangent 1 0 0.
   Side light from +X, medium Cornell box.
   What to see: elongated gold specular streak on the brushed metal plate;
   highlight shape set by anisotropic Ward lobe along tangent X.

2. ward_aniso_B.bmp
   Scene: testcases/scene_ward_aniso_B.txt
   Same geometry/light/camera as A; ONLY tangent changed to 0 1 0 (90 deg).
   What to see: streak direction rotated ~90 deg vs image A — proves anisotropy
   follows material tangent, not just roughness.

3. ward_aniso_showcase.bmp (optional report figure)
   Scene: testcases/scene_ward_aniso_showcase.txt
   Three vertical gold panels:
   - LEFT: isotropic alpha 0.12 / 0.12 — near-round highlight (aspect ~1.4 tall)
   - CENTER: aniso alpha 0.04 / 0.45, tangent 1 0 0 — moderate oval
   - RIGHT: aniso alpha 0.04 / 0.45, tangent 0 1 0 — wide horizontal streak
     (aspect ~2.4 wide)
   Use fixed-tangent vertical planes (not spheres) so streak direction is stable.

512 SPP (optional):
  replace 256 with 512 in commands above.
