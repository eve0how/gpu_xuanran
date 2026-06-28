#ifndef BVH_BUILDER_HPP
#define BVH_BUILDER_HPP

#include "cuda_types.h"

#include <vector>

void buildBVH(const std::vector<GpuTriangle> &triangles, std::vector<GpuBVHNode> &outNodes,
              std::vector<GpuTriangle> &outTriangles);

#endif
