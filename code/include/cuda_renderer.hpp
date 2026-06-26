#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "scene_parser.hpp"
#include "image.hpp"
#include "raytracer.hpp"

bool cudaAvailable();

bool renderWithCuda(const SceneParser &scene, Image &image, RenderMode mode, int spp,
                    bool dispersion, double &renderSec, int trainSpp = 0);

void freeCudaSceneCache();

#endif
