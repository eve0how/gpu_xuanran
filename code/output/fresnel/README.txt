Fresnel Schlick Bonus Renders (256 SPP, path_nee, CUDA, gamma)
=============================================================

Build:
  cd /data/PA1-2/code/build && cmake .. && make -j

Batch render + analysis:
  cd /data/PA1-2/code && bash scripts/fresnel_render.sh

Individual commands:
  ./build/PA1-2 testcases/scene_fresnel_cornell_compare.txt \
      output/fresnel/fresnel_cornell_compare.bmp path_nee 256 gamma cuda

  ./build/PA1-2 testcases/scene_fresnel_cornell_water_glass.txt \
      output/fresnel/fresnel_cornell_water_glass.bmp path_nee 256 gamma cuda

  ./build/PA1-2 testcases/scene_fresnel_grazing_topdown.txt \
      output/fresnel/fresnel_grazing_topdown.bmp path_nee 256 gamma cuda

  ./build/PA1-2 testcases/scene_fresnel_grazing_low.txt \
      output/fresnel/fresnel_grazing_low.bmp path_nee 256 gamma cuda

Pixel analysis:
  python3 scripts/analyze_fresnel_regions.py output/fresnel/*.png
  python3 scripts/fresnel_figures.py

----------------------------------------------------------------------
1. fresnel_cornell_compare.bmp / .png / _labeled.png
   Scene: testcases/scene_fresnel_cornell_compare.txt
   Params: camera (0,0.50,3.8)->(0,0,-1); spheres at (-0.55,0.50,0) and (0.55,0.50,0),
            r=0.28 (gap 0.54); IOR 1.50 white; LEFT noFresnel, RIGHT Fresnel ON.
   What to see:
   - LEFT: center sees distorted back wall (more transparent); weak side reflections.
   - RIGHT: center still transparent; brighter red/green wall reflections on rim/sides.
   - Pixel check (64+ SPP): LEFT center L2 to back wall ~0.02–0.05; RIGHT ~0.05–0.10;
     RIGHT rim G channel higher than LEFT on green-wall side.

2. fresnel_cornell_water_glass.bmp / .png / _labeled.png
   Scene: testcases/scene_fresnel_cornell_water_glass.txt
   Params: same camera; LEFT water IOR 1.33, refractColor (0.30,0.55,1.0), r=0.32;
            RIGHT glass IOR 1.60 white, r=0.26; both Fresnel ON.
   What to see:
   - LEFT: strong cyan/blue tint (B/R > 1.1), back wall visible through center.
   - RIGHT: clear neutral glass, sharper Fresnel rim, tighter/brighter caustics.

2b. fresnel_debug_transmit.bmp
   Scene: testcases/scene_fresnel_debug_transmit.txt
   Params: single centered noFresnel sphere (0,0.50,0) r=0.28; camera (0,0.50,2.5).
   Sanity check: sphere center L2 to back wall < 0.03 confirms refraction path.

3. fresnel_grazing_topdown.bmp / fresnel_grazing_low.bmp (+ labeled + compare)
   Scenes: scene_fresnel_grazing_topdown.txt / scene_fresnel_grazing_low.txt
   Params: glass floor IOR 1.50 Fresnel ON; red sphere (0.9, 0.1, 0.1) on floor;
            topdown camera (0,5,0.01)->(0,-1,0); low camera (0,0.08,4.5)->(0,0,-1).
   What to see:
   - TOPDOWN: floor nearly transparent; weak/no red reflection in floor.
   - LOW: floor acts as mirror; bright red sphere reflection on floor.
   - Side-by-side: output/fresnel/fresnel_grazing_compare.png

512 SPP (optional, less noise):
  append 512 instead of 256 in commands above.
