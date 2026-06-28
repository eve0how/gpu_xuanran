#ifndef BVH_BUILDER_HPP
#define BVH_BUILDER_HPP

// 文件说明：CPU 端 BVH 构建，输出扁平 GpuBVHNode 数组。
// 原创性声明：参考已有代码（PBRT BVH 构建思路），buildBVH 接口独立实现。

#include "cuda_types.h"

#include <vector>

// Flattened SAH-style BVH over triangle soup; outputs GpuBVHNode array for GPU traversal.
void buildBVH(const std::vector<GpuTriangle> &triangles, std::vector<GpuBVHNode> &outNodes,
              std::vector<GpuTriangle> &outTriangles);

#endif
