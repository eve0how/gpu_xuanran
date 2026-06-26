#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

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
constexpr int kGuideGridRes = 16;
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
    float x, y, z;
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
