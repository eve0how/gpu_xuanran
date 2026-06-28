#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

// 文件说明：CUDA 渲染器对外接口，封装场景上传与内核调度。
// 原创性声明：参考已有代码（PA1 CUDA 渲染入口模式），接口声明独立实现。

#include "scene_parser.hpp"
#include "image.hpp"
#include "raytracer.hpp"

bool cudaAvailable();

void setGpuSceneBuildUseBVH(bool use);

// Host-side CUDA render entry: uploads flattened scene and launches path/Whitted kernels.
bool renderWithCuda(const SceneParser &scene, Image &image, RenderMode mode, int spp,
                    bool dispersion, double &renderSec, int trainSpp = 0, bool useBvh = true);

void freeCudaSceneCache();

#endif
