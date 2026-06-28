#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

// 文件说明：GPU 端场景 SOA 布局、BVH 节点与渲染模式枚举。
// 原创性声明：参考已有代码（CUDA 路径追踪数据结构惯例），
// GpuBVHNode 与 GpuSceneHost 字段布局按作业需求独立实现。

#include <cstdint>

enum GpuMatType : int {
    GPU_MAT_DIFFUSE = 0,
    GPU_MAT_REFLECT = 1,
    GPU_MAT_REFRACT = 2,
    GPU_MAT_EMISSIVE = 3,
    GPU_MAT_GLOSSY = 4
};

enum GpuRenderMode : int {
    GPU_WHITTED = 0,
    GPU_PATH = 1,
    GPU_PATH_NEE = 2,
    GPU_PATH_MIS = 3,
    GPU_PATH_GUIDING = 4
};

// Practical path guiding: 3D grid over scene AABB, lat-long directional histogram per cell.
// Path guiding grid resolution — 8³ cells × 16×16 lat-long bins per cell.
constexpr int kGuideGridRes = 8;
constexpr int kGuideThetaBins = 16;
constexpr int kGuidePhiBins = 16;
constexpr int kGuideBinsPerCell = kGuideThetaBins * kGuidePhiBins;
constexpr int kGuideNumCells = kGuideGridRes * kGuideGridRes * kGuideGridRes;

struct GpuGuidingGrid {
    float bboxMin[3];
    float bboxMax[3];
    int res;
    int thetaBins;
    int phiBins;
    int binsPerCell;
    float *weights;
};

struct GpuVec3 {
    float x;
    float y;
    float z;
};

struct GpuMaterial {
    int type;
    float diffuse[3];
    float specular[3];
    float emission[3];
    float ior;
    float roughness;
    float f0[3];
    float dispersionDelta;
    float shininess;
};

struct GpuSphere {
    float center[3];
    float radius;
    int matId;
};

struct GpuPlane {
    float normal[3];
    float offset;
    int matId;
};

struct GpuTriangle {
    float v0[3];
    float v1[3];
    float v2[3];
    int matId;
};

struct AABB {
    float bmin[3];
    float bmax[3];
};

// 32-byte aligned BVH node: leaf uses leftChild as first triangle offset in bvhTriangles.
struct alignas(32) GpuBVHNode {
    float bboxMin[3];
    float bboxMax[3];
    int leftChild;
    int rightChild;
    int primitiveCount;
    int _pad;
};

struct GpuAreaLight {
    float v0[3];
    float v1[3];
    float v2[3];
    float color[3];
};

struct GpuPointLight {
    float pos[3];
    float color[3];
};

struct GpuDirectionalLight {
    float direction[3];
    float color[3];
};

struct GpuCamera {
    float center[3];
    float direction[3];
    float horizontal[3];
    float up[3];
    float cx, cy, fx, fy;
    float bg[3];
};

struct GpuSceneHost {
    GpuMaterial *materials = nullptr;
    int numMaterials = 0;
    GpuSphere *spheres = nullptr;
    int numSpheres = 0;
    GpuPlane *planes = nullptr;
    int numPlanes = 0;
    GpuTriangle *triangles = nullptr;
    int numTriangles = 0;
    GpuBVHNode *bvhNodes = nullptr;
    int numBVHNodes = 0;
    GpuTriangle *bvhTriangles = nullptr;
    int numBVHTriangles = 0;
    int bvhRootIndex = 0;
    GpuAreaLight *areaLights = nullptr;
    int numAreaLights = 0;
    GpuPointLight *pointLights = nullptr;
    int numPointLights = 0;
    GpuDirectionalLight *directionalLights = nullptr;
    int numDirectionalLights = 0;
    GpuCamera camera{};
    float bboxMin[3]{};
    float bboxMax[3]{};
    bool hasBbox = false;
};

#endif
