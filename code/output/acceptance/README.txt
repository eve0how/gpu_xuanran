PA1-2 Acceptance Render Manifest
================================
Generated: 2026-06-26 (dispersion exit-split fix + scene redesign)
Resolution: 1024x1024 (all PNGs > 640x480)
Backend: CUDA
Format: lossless PNG (converted from BMP, optimize=False)

Required (必做)
---------------
whitted.png
  ./build/PA1-2 testcases/scene_whitted.txt output/acceptance/whitted.bmp whitted cuda
  EXPECT: Cornell-style box, Phong spheres + glass cube, crisp edges, no MC noise.

path_no_nee.png
  ./build/PA1-2 testcases/scene_path.txt output/acceptance/path_no_nee.bmp path 64 gamma cuda
  EXPECT: Same scene as path_nee but much darker, grainy, soft shadows missing.

path_nee.png  (256 spp)
  ./build/PA1-2 testcases/scene_path.txt output/acceptance/path_nee.bmp path_nee 256 gamma cuda
  EXPECT: Bright Cornell box, soft penumbra under spheres, smooth floor color bleeding.

glossy.png
  ./build/PA1-2 testcases/scene_glossy.txt output/acceptance/glossy.bmp path_nee 64 gamma cuda
  EXPECT: Cook-Torrance glossy spheres with specular highlights (point light only).

Gamma Correction Before/After
-------------------------------
gamma_before.png  (linear, no gamma)
  ./build/PA1-2 testcases/scene_whitted.txt output/acceptance/gamma_before.bmp whitted cuda

gamma_after.png  (gamma correction on)
  ./build/PA1-2 testcases/scene_whitted.txt output/acceptance/gamma_after.bmp whitted gamma cuda

MIS Three-Way Comparison (scene_mis_demo.txt, 32 spp)
-----------------------------------------------------
Scene: tiny ceiling area light (0.07 m), five near-mirror glossy spheres
(roughness 0.004–0.005), bright emission 520. path_nee counts indirect
emissive hits at full strength (fireflies); path_mis applies power-heuristic MIS.

mis_compare_brdf_32.png  (path mode — BRDF-only, NO NEE)
  ./build/PA1-2 testcases/scene_mis_demo.txt output/acceptance/mis_compare_brdf_32.bmp path 32 gamma cuda
  EXPECT: Very dark overall; heavy Monte Carlo noise.

mis_compare_nee_32.png  (path_nee — NEE, no MIS weights)
  ./build/PA1-2 testcases/scene_mis_demo.txt output/acceptance/mis_compare_nee_32.bmp path_nee 32 gamma cuda
  EXPECT: Many white/colored firefly speckles on walls, ceiling, glossy spheres.

mis_compare_mis_32.png  (path_mis — NEE + Power-heuristic MIS)
  ./build/PA1-2 testcases/scene_mis_demo.txt output/acceptance/mis_compare_mis_32.bmp path_mis 32 gamma cuda
  EXPECT: Same layout, clearly smoother — no scattered firefly dots.

  Verified 2026-06-26: 2.2% pixels differ by >1 DN; 1.4% by >10 DN;
  NEE-only hot speckles (NEE>200, MIS<150): ~1555; NEE brighter by >15 DN: ~11343.

Dispersion Before/After (scene_prism.txt — triangular prism mesh)
-----------------------------------------------------------------
Neutral gray box (ceiling + left/right walls 0.60), ceiling slit
(0.15 m — widened from 0.10 m for easier caustic sampling), white
back wall at z=-1 as projection screen, glass prism (mesh/prism.obj)
with dispersionDelta 0.10. Light path: slit → prism → back screen.
Dispersion splits RGB only on glass EXIT (air boundary).

dispersion_before.png  (no dispersion — single IOR, 1024 spp)
  ./build/PA1-2 testcases/scene_prism.txt output/acceptance/dispersion_before.bmp path_nee 1024 gamma cuda
  EXPECT: White spotlight on back wall through prism; no color fringing.

dispersion_after.png  (RGB channel IOR split on exit, 1024 spp)
  ./build/PA1-2 testcases/scene_prism.txt output/acceptance/dispersion_after.bmp path_nee 1024 gamma dispersion cuda
  EXPECT: Red/green/blue rainbow bands on white back wall.

  Verified 2026-06-26: back-wall ROI (x:320-704, y:90-380) lum_std
  ~20 DN at 1024 spp (0.15 m slit). Caustics + dispersion remain noisy
  vs Cornell scenes — see Notes on variance.

Anti-Aliasing Before/After (jittered supersampling, Whitted mode)
-------------------------------------------------------------------
aa_before.png  (SPP=1)
  ./build/PA1-2 testcases/scene_whitted.txt output/acceptance/aa_before.bmp whitted 1 cuda

aa_after.png  (SPP=16)
  ./build/PA1-2 testcases/scene_whitted.txt output/acceptance/aa_after.bmp whitted 16 cuda

Notes
-----
- Dispersion: channelIor uses base±delta/2; scaleDispAttenuation keeps
  single R/G/B wavelength per sub-path; split occurs on EXIT refraction only
  (not entry), matching classic prism demo physics.
- Dispersion noise: back-wall caustics use refractive paths (slit→prism→wall);
  NEE shadow rays treat glass as opaque (isSegmentOccluded), so the wall
  cannot directly sample the ceiling slit through the prism — only indirect
  specular-diffuse chains contribute. Narrow lights + caustics need very
  high SPP (1024+); 0.15 m slit helps without changing physics model.
- CUDA Whitted mode now also supports dispersion (exit split).
- MIS: direct NEE no longer clamped; indirect emissive enabled for path_nee
  (unweighted) vs path_mis (MIS-weighted) to show firefly contrast.
- scene_dispersion.txt aliases scene_prism.txt.
