#include "cuda_types.h"
#include "cuda_renderer.hpp"
#include "cuda_device.hpp"
#include "scene_parser.hpp"
#include "image.hpp"
#include "camera.hpp"
#include "raytracer.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

extern GpuSceneHost buildGpuSceneHost(const SceneParser &scene);

namespace {

constexpr float kRayEps = 1e-4f;
constexpr float kOriginOffset = 1e-3f;
constexpr float kRefractOriginOffset = 2e-3f;
constexpr float kRefractRayTMin = 2e-3f;
constexpr float kShadowEps = 1e-3f;
constexpr int kMaxDepth = 12;
constexpr int kRrStartDepth = 8;
constexpr float kRrMinSurvival = 0.15f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kRadianceClamp = 100.0f;

struct GpuSceneDevice {
    const GpuMaterial *materials;
    int numMaterials;
    const GpuSphere *spheres;
    int numSpheres;
    const GpuPlane *planes;
    int numPlanes;
    const GpuTriangle *triangles;
    int numTriangles;
    const GpuAreaLight *areaLights;
    int numAreaLights;
    const GpuPointLight *pointLights;
    int numPointLights;
    const GpuDirectionalLight *directionalLights;
    int numDirectionalLights;
    GpuCamera camera;
};

struct HitRec {
    float t;
    int matId;
    float n[3];
    bool hit;
};

__device__ inline float3 make3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ inline float3 load3(const float v[3]) {
    return make_float3(v[0], v[1], v[2]);
}

__device__ inline void store3(float out[3], float3 v) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

__device__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 add3(float3 a, float3 b) {
    return make3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 sub3(float3 a, float3 b) {
    return make3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 mul3(float3 a, float s) {
    return make3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float3 mul3v(float3 a, float3 b) {
    return make3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float len3(float3 a) {
    return sqrtf(dot3(a, a));
}

__device__ inline float3 norm3(float3 a) {
    float l = len3(a);
    if (l < 1e-12f) {
        return make3(0.0f, 1.0f, 0.0f);
    }
    return mul3(a, 1.0f / l);
}

__device__ inline float3 cross3(float3 a, float3 b) {
    return make3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ inline float luminance3(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ inline float3 clampRadiance3(float3 c) {
    float lum = luminance3(c);
    if (lum > kRadianceClamp) {
        return mul3(c, kRadianceClamp / lum);
    }
    return c;
}

__device__ inline float gpuUniform(curandState &state) {
    return curand_uniform(&state);
}

__device__ inline float3 faceNormal(float3 dir, float3 geomN) {
    float3 n = norm3(geomN);
    if (dot3(dir, n) > 0.0f) {
        n = mul3(n, -1.0f);
    }
    return n;
}

// Matches Material::faceShadingNormal / getShadingNormal on CPU.
__device__ inline float3 shadingNormal(float3 viewDir, float3 geomN) {
    float3 n = norm3(geomN);
    if (dot3(viewDir, n) < 0.0f) {
        n = mul3(n, -1.0f);
    }
    return n;
}

__device__ inline float3 offsetAlongNormal(float3 p, float3 n, float eps) {
    return add3(p, mul3(norm3(n), eps));
}

__device__ inline float3 offsetAlongRay(float3 p, float3 dir, float eps) {
    return add3(p, mul3(norm3(dir), eps));
}

__device__ inline bool rayTriangle(const float3 orig, const float3 dir,
                                   const float3 v0, const float3 v1, const float3 v2,
                                   float &tOut, float &uOut, float &vOut) {
    float3 e1 = sub3(v1, v0);
    float3 e2 = sub3(v2, v0);
    float3 pvec = cross3(dir, e2);
    float det = dot3(e1, pvec);
    if (fabsf(det) < 1e-8f) {
        return false;
    }
    float invDet = 1.0f / det;
    float3 tvec = sub3(orig, v0);
    uOut = dot3(tvec, pvec) * invDet;
    if (uOut < 0.0f || uOut > 1.0f) {
        return false;
    }
    float3 qvec = cross3(tvec, e1);
    vOut = dot3(dir, qvec) * invDet;
    if (vOut < 0.0f || uOut + vOut > 1.0f) {
        return false;
    }
    tOut = dot3(e2, qvec) * invDet;
    return tOut > kRayEps;
}

__device__ bool intersectScene(const GpuSceneDevice &scene, float3 orig, float3 dir,
                               float tmin, HitRec &best) {
    best.hit = false;
    best.t = 1e30f;

    for (int i = 0; i < scene.numSpheres; ++i) {
        const GpuSphere &s = scene.spheres[i];
        float3 center = load3(s.center);
        float3 oc = sub3(orig, center);
        float a = dot3(dir, dir);
        float b = 2.0f * dot3(oc, dir);
        float c = dot3(oc, oc) - s.radius * s.radius;
        float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) {
            continue;
        }
        float sqrtDisc = sqrtf(disc);
        float t0 = (-b - sqrtDisc) / (2.0f * a);
        float t1 = (-b + sqrtDisc) / (2.0f * a);
        float t = -1.0f;
        if (t0 > tmin) {
            t = t0;
        } else if (t1 > tmin) {
            t = t1;
        }
        if (t > 0.0f && t < best.t) {
            best.t = t;
            best.matId = s.matId;
            float3 p = add3(orig, mul3(dir, t));
            store3(best.n, sub3(p, center));
            best.hit = true;
        }
    }

    for (int i = 0; i < scene.numPlanes; ++i) {
        const GpuPlane &pl = scene.planes[i];
        float3 n = load3(pl.normal);
        float denom = dot3(n, dir);
        if (fabsf(denom) < 1e-6f) {
            continue;
        }
        float t = (pl.offset - dot3(n, orig)) / denom;
        if (t > tmin && t < best.t) {
            best.t = t;
            best.matId = pl.matId;
            store3(best.n, n);
            best.hit = true;
        }
    }

    for (int i = 0; i < scene.numTriangles; ++i) {
        const GpuTriangle &tri = scene.triangles[i];
        float t = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (!rayTriangle(orig, dir, load3(tri.v0), load3(tri.v1), load3(tri.v2), t, u, v)) {
            continue;
        }
        if (t > tmin && t < best.t) {
            best.t = t;
            best.matId = tri.matId;
            float3 e1 = sub3(load3(tri.v1), load3(tri.v0));
            float3 e2 = sub3(load3(tri.v2), load3(tri.v0));
            store3(best.n, cross3(e1, e2));
            best.hit = true;
        }
    }

    return best.hit;
}

__device__ float triangleArea(float3 v0, float3 v1, float3 v2) {
    return 0.5f * len3(cross3(sub3(v1, v0), sub3(v2, v0)));
}

__device__ float3 computeRefractDirection(float3 D, float3 geomN, float ior) {
    float etai = 1.0f;
    float etat = ior;
    float3 n = geomN;
    float cosTheta = dot3(D, n);
    if (cosTheta > 0.0f) {
        etai = ior;
        etat = 1.0f;
        n = mul3(n, -1.0f);
        cosTheta = -cosTheta;
    }
    float eta = etai / etat;
    float k = 1.0f - eta * eta * (1.0f - cosTheta * cosTheta);
    if (k < 0.0f) {
        return norm3(sub3(D, mul3(n, 2.0f * dot3(D, n))));
    }
    return norm3(add3(mul3(D, eta), mul3(n, eta * cosTheta - sqrtf(k))));
}

__device__ void buildBasis(float3 n, float3 &tangent, float3 &bitangent) {
    if (fabsf(n.x) > fabsf(n.y)) {
        tangent = norm3(make3(-n.z, 0.0f, n.x));
    } else {
        tangent = norm3(make3(0.0f, n.z, -n.y));
    }
    bitangent = cross3(n, tangent);
}

__device__ float3 sampleCosineHemisphere(float3 normal, float &pdf, curandState &rng) {
    float u1 = gpuUniform(rng);
    float u2 = gpuUniform(rng);
    float phi = 2.0f * kPi * u1;
    float cosTheta = sqrtf(u2);
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float3 tangent, bitangent;
    buildBasis(normal, tangent, bitangent);
    pdf = cosTheta / kPi;
    return add3(add3(mul3(tangent, cosf(phi) * sinTheta), mul3(bitangent, sinf(phi) * sinTheta)),
                  mul3(normal, cosTheta));
}

__device__ float beckmannD(float3 n, float3 h, float roughness) {
    float cosTheta = fmaxf(0.001f, dot3(n, h));
    float cosTheta2 = cosTheta * cosTheta;
    float tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
    float m2 = roughness * roughness;
    return expf(-tanTheta2 / m2) / (kPi * m2 * cosTheta2 * cosTheta2);
}

__device__ float cookTorranceG1(float3 n, float3 w, float3 h) {
    float nw = fmaxf(0.0f, dot3(n, w));
    float nh = fmaxf(0.0f, dot3(n, h));
    float wh = fmaxf(0.001f, dot3(w, h));
    return fminf(1.0f, 2.0f * nh * nw / wh);
}

__device__ float3 schlickF(float3 f0, float cosTheta) {
    float f = powf(1.0f - cosTheta, 5.0f);
    return add3(f0, mul3(sub3(make3(1, 1, 1), f0), f));
}

__device__ float3 evalSpecular(float3 n, float3 wo, float3 wi, float3 ks, float roughness, float3 f0) {
    float nl = dot3(n, wi);
    float nv = dot3(n, wo);
    if (nl <= 0.0f || nv <= 0.0f) {
        return make3(0, 0, 0);
    }
    float3 h = norm3(add3(wi, wo));
    float nh = dot3(n, h);
    if (nh <= 0.0f) {
        return make3(0, 0, 0);
    }
    float D = beckmannD(n, h, roughness);
    float G = cookTorranceG1(n, wo, h) * cookTorranceG1(n, wi, h);
    float3 F = schlickF(f0, fmaxf(0.0f, dot3(h, wo)));
    float scalar = (D * G) / (4.0f * nl * nv);
    return mul3v(mul3(F, scalar), ks);
}

__device__ float3 evalGlossy(float3 n, float3 wo, float3 wi, const GpuMaterial &mat) {
    float nl = dot3(n, wi);
    float nv = dot3(n, wo);
    if (nl <= 0.0f || nv <= 0.0f) {
        return make3(0, 0, 0);
    }
    float3 kd = load3(mat.diffuse);
    float3 diffuse = mul3(kd, 1.0f / kPi);
    float3 spec = evalSpecular(n, wo, wi, load3(mat.specular), mat.roughness, load3(mat.f0));
    return add3(diffuse, spec);
}

__device__ float pdfDiffuse(float cosO) {
    return cosO / kPi;
}

__device__ float pdfGlossy(float3 n, float3 wo, float3 wi, const GpuMaterial &mat) {
    float cosO = fmaxf(0.0f, dot3(n, wi));
    if (cosO <= 0.0f) {
        return 0.0f;
    }
    float3 kd = load3(mat.diffuse);
    float3 ks = load3(mat.specular);
    float kdLum = luminance3(kd);
    float ksLum = luminance3(ks);
    bool isMetal = kdLum < 0.01f;
    float specProb = isMetal ? 1.0f : ksLum / fmaxf(1e-4f, kdLum + ksLum);
    float pdfDiff = (1.0f - specProb) * cosO / kPi;
    float3 h = norm3(add3(wi, wo));
    float nh = fmaxf(0.0f, dot3(n, h));
    if (nh <= 0.0f) {
        return pdfDiff;
    }
    float D = beckmannD(n, h, mat.roughness);
    float pdfH = D * nh;
    float vh = fmaxf(0.001f, dot3(wo, h));
    float pdfSpec = specProb * pdfH / (4.0f * vh);
    return pdfDiff + pdfSpec;
}

__device__ float pdfAreaLightDir(const GpuAreaLight &area, float3 hitPoint, float3 wi) {
    float t = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    if (!rayTriangle(hitPoint, wi, load3(area.v0), load3(area.v1), load3(area.v2), t, u, v)) {
        return 0.0f;
    }
    float3 lightN = norm3(cross3(sub3(load3(area.v1), load3(area.v0)), sub3(load3(area.v2), load3(area.v0))));
    float cosL = dot3(lightN, mul3(wi, -1.0f));
    if (cosL <= 0.0f) {
        return 0.0f;
    }
    float lightArea = triangleArea(load3(area.v0), load3(area.v1), load3(area.v2));
    return t * t / (lightArea * cosL);
}

__device__ float computeAreaLightPdf(const GpuSceneDevice &scene, float3 hitPoint, float3 wi) {
    float pdf = 0.0f;
    for (int i = 0; i < scene.numAreaLights; ++i) {
        float p = pdfAreaLightDir(scene.areaLights[i], hitPoint, wi);
        if (p > 0.0f) {
            pdf += p;
        }
    }
    return pdf;
}

__device__ float misPowerDenom(float pdfA, float pdfB) {
    return pdfA * pdfA + pdfB * pdfB;
}

__device__ float misWeightPower(float pdf, float pdfA, float pdfB) {
    float denom = misPowerDenom(pdfA, pdfB);
    if (denom < 1e-8f) {
        return 0.0f;
    }
    return pdf * pdf / denom;
}

__device__ float misPowerDenom3(float pdfA, float pdfB, float pdfC) {
    return pdfA * pdfA + pdfB * pdfB + pdfC * pdfC;
}

__device__ float misWeightPower3(float pdf, float pdfA, float pdfB, float pdfC) {
    float denom = misPowerDenom3(pdfA, pdfB, pdfC);
    if (denom < 1e-8f) {
        return 0.0f;
    }
    return pdf * pdf / denom;
}

// --- Path guiding (simplified Practical Path Guiding: 3D grid + lat-long bins) ---

__device__ int guideCellIndex(const GpuGuidingGrid &grid, float3 pos) {
    float3 bmin = load3(grid.bboxMin);
    float3 bmax = load3(grid.bboxMax);
    float3 extent = sub3(bmax, bmin);
    float ex = fmaxf(extent.x, 1e-4f);
    float ey = fmaxf(extent.y, 1e-4f);
    float ez = fmaxf(extent.z, 1e-4f);
    float u = fminf(fmaxf((pos.x - bmin.x) / ex, 0.0f), 0.999999f);
    float v = fminf(fmaxf((pos.y - bmin.y) / ey, 0.0f), 0.999999f);
    float w = fminf(fmaxf((pos.z - bmin.z) / ez, 0.0f), 0.999999f);
    int res = grid.res;
    int ix = static_cast<int>(u * res);
    int iy = static_cast<int>(v * res);
    int iz = static_cast<int>(w * res);
    return ix + iy * res + iz * res * res;
}

__device__ void wiToGuideBins(float3 N, float3 wi, int thetaBins, int phiBins, int &tBin, int &pBin) {
    float cosTheta = fmaxf(0.0f, fminf(1.0f, dot3(N, wi)));
    float theta = acosf(cosTheta);
    float3 tangent, bitangent;
    buildBasis(N, tangent, bitangent);
    float x = dot3(wi, tangent);
    float y = dot3(wi, bitangent);
    float phi = atan2f(y, x);
    if (phi < 0.0f) {
        phi += 2.0f * kPi;
    }
    float halfPi = 0.5f * kPi;
    tBin = fminf(thetaBins - 1, static_cast<int>(theta / halfPi * thetaBins));
    pBin = fminf(phiBins - 1, static_cast<int>(phi / (2.0f * kPi) * phiBins));
}

__device__ int guideDirBin(const GpuGuidingGrid &grid, float3 N, float3 wi) {
    int tBin = 0;
    int pBin = 0;
    wiToGuideBins(N, wi, grid.thetaBins, grid.phiBins, tBin, pBin);
    return tBin * grid.phiBins + pBin;
}

__device__ float guideBinSolidAngle(const GpuGuidingGrid &grid, int tBin) {
    float halfPi = 0.5f * kPi;
    float thetaMin = tBin * halfPi / grid.thetaBins;
    float thetaMax = (tBin + 1) * halfPi / grid.thetaBins;
    return (cosf(thetaMin) - cosf(thetaMax)) * (2.0f * kPi / grid.phiBins);
}

__device__ float guideCellSum(const GpuGuidingGrid &grid, int cell) {
    float sum = 0.0f;
    int base = cell * grid.binsPerCell;
    for (int b = 0; b < grid.binsPerCell; ++b) {
        sum += grid.weights[base + b];
    }
    return sum;
}

__device__ void guideDeposit(const GpuGuidingGrid &grid, float3 pos, float3 N, float3 wi, float weight) {
    if (weight < 1e-10f || grid.weights == nullptr) {
        return;
    }
    float cosIn = dot3(N, wi);
    if (cosIn <= 0.0f) {
        return;
    }
    int cell = guideCellIndex(grid, pos);
    int bin = guideDirBin(grid, N, wi);
    atomicAdd(&grid.weights[cell * grid.binsPerCell + bin], weight * cosIn);
}

__device__ float evalGuidingPdf(const GpuGuidingGrid &grid, float3 pos, float3 N, float3 wi) {
    if (grid.weights == nullptr || dot3(N, wi) <= 0.0f) {
        return 0.0f;
    }
    int cell = guideCellIndex(grid, pos);
    float sum = guideCellSum(grid, cell);
    if (sum < 1e-8f) {
        return 0.0f;
    }
    int tBin = 0;
    int pBin = 0;
    wiToGuideBins(N, wi, grid.thetaBins, grid.phiBins, tBin, pBin);
    int bin = tBin * grid.phiBins + pBin;
    float w = grid.weights[cell * grid.binsPerCell + bin];
    if (w <= 0.0f) {
        return 0.0f;
    }
    float binOmega = guideBinSolidAngle(grid, tBin);
    if (binOmega < 1e-8f) {
        return 0.0f;
    }
    return w / sum / binOmega;
}

__device__ float3 dirFromGuideBin(float3 N, int tBin, int pBin, float u1, float u2,
                                  const GpuGuidingGrid &grid) {
    float halfPi = 0.5f * kPi;
    float thetaMin = tBin * halfPi / grid.thetaBins;
    float thetaMax = (tBin + 1) * halfPi / grid.thetaBins;
    float cosThetaMin = cosf(thetaMin);
    float cosThetaMax = cosf(thetaMax);
    float cosTheta = cosThetaMin + u1 * (cosThetaMax - cosThetaMin);
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = (pBin + u2) * (2.0f * kPi / grid.phiBins);
    float3 tangent, bitangent;
    buildBasis(N, tangent, bitangent);
    return norm3(add3(add3(mul3(tangent, cosf(phi) * sinTheta), mul3(bitangent, sinf(phi) * sinTheta)),
                      mul3(N, cosTheta)));
}


__device__ float3 sampleGuidingDir(const GpuGuidingGrid &grid, float3 pos, float3 N, float &pdf,
                                   curandState &rng);

__device__ bool guideCellHasData(const GpuGuidingGrid &grid, float3 pos) {
    return guideCellSum(grid, guideCellIndex(grid, pos)) >= 1e-8f;
}

constexpr float kGuideMisProb = 0.5f;

// Balance-heuristic one-sample MIS: pdf_total = 0.5*pdf_brdf + 0.5*pdf_guide.
__device__ bool sampleIndirectWithGuide(const GpuGuidingGrid &grid, float3 pos, float3 N,
                                        float3 &wi, float &pdfTotal, curandState &rng) {
    if (!guideCellHasData(grid, pos)) {
        return false;
    }
    int strat = gpuUniform(rng) < kGuideMisProb ? 1 : 0;
    if (strat == 0) {
        float unused = 0.0f;
        wi = sampleCosineHemisphere(N, unused, rng);
    } else {
        float unused = 0.0f;
        wi = sampleGuidingDir(grid, pos, N, unused, rng);
    }
    if (dot3(N, wi) <= 0.0f) {
        return false;
    }
    float cosO = fmaxf(0.0f, dot3(N, wi));
    float pdfBrdf = pdfDiffuse(cosO);
    float pdfGuide = evalGuidingPdf(grid, pos, N, wi);
    pdfTotal = kGuideMisProb * pdfBrdf + kGuideMisProb * pdfGuide;
    return pdfTotal >= 1e-8f;
}

__device__ float3 sampleGuidingDir(const GpuGuidingGrid &grid, float3 pos, float3 N, float &pdf,
                                   curandState &rng) {
    int cell = guideCellIndex(grid, pos);
    float sum = guideCellSum(grid, cell);
    if (sum < 1e-8f) {
        pdf = 0.0f;
        return make3(0, 0, 0);
    }
    float u = gpuUniform(rng) * sum;
    int base = cell * grid.binsPerCell;
    int chosen = base;
    float accum = 0.0f;
    for (int b = 0; b < grid.binsPerCell; ++b) {
        accum += grid.weights[base + b];
        if (u <= accum) {
            chosen = base + b;
            break;
        }
    }
    int binLocal = chosen - base;
    int tBin = binLocal / grid.phiBins;
    int pBin = binLocal % grid.phiBins;
    float u1 = gpuUniform(rng);
    float u2 = gpuUniform(rng);
    float3 wi = dirFromGuideBin(N, tBin, pBin, u1, u2, grid);
    pdf = evalGuidingPdf(grid, pos, N, wi);
    return wi;
}

__device__ bool useGuidingMode(int mode) {
    return mode == GPU_PATH_GUIDING;
}

__device__ float channelIor(float baseIor, float delta, int channel) {
    if (channel == 0) {
        return baseIor - delta * 0.5f;
    }
    if (channel == 2) {
        return baseIor + delta * 0.5f;
    }
    return baseIor;
}

// After exit split each child keeps one RGB channel — prevents re-mixing to white inside glass.
__device__ float3 scaleDispAttenuation(float3 throughput, float3 attenColor, int dispChannel) {
    if (dispChannel < 0) {
        return mul3v(throughput, attenColor);
    }
    if (dispChannel == 0) {
        return make3(throughput.x * attenColor.x, 0.0f, 0.0f);
    }
    if (dispChannel == 1) {
        return make3(0.0f, throughput.y * attenColor.y, 0.0f);
    }
    return make3(0.0f, 0.0f, throughput.z * attenColor.z);
}

__device__ bool isSegmentOccluded(const GpuSceneDevice &scene, float3 from, float3 to, float3 N) {
    float3 dir = sub3(to, from);
    float dist = len3(dir);
    if (dist < kRayEps) {
        return false;
    }
    dir = mul3(dir, 1.0f / dist);
    float3 shadowN = N;
    if (dot3(shadowN, dir) < 0.0f) {
        shadowN = mul3(shadowN, -1.0f);
    }
    float3 shadowOrigin = add3(from, mul3(shadowN, kShadowEps));
    float segDist = len3(sub3(to, shadowOrigin));
    HitRec hit;
    if (!intersectScene(scene, shadowOrigin, dir, kRayEps, hit)) {
        return false;
    }
    if (hit.t >= segDist - kShadowEps) {
        return false;
    }
    const GpuMaterial &blocker = scene.materials[hit.matId < 0 || hit.matId >= scene.numMaterials ? 0 : hit.matId];
    if (blocker.type == GPU_MAT_EMISSIVE) {
        return false;
    }
    return true;
}

__device__ float3 sampleTriangle(float3 v0, float3 v1, float3 v2, curandState &rng) {
    float u = gpuUniform(rng);
    float v = gpuUniform(rng);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return add3(v0, add3(mul3(sub3(v1, v0), u), mul3(sub3(v2, v0), v)));
}

__device__ bool sampleAreaLightDiffuse(const GpuSceneDevice &scene, const GpuAreaLight &area,
                                       float3 hitPoint, float3 N, float3 albedo, bool useMis,
                                       float3 &contrib, curandState &rng) {
    float3 lightPoint = sampleTriangle(load3(area.v0), load3(area.v1), load3(area.v2), rng);
    float3 wi = sub3(lightPoint, hitPoint);
    float dist2 = dot3(wi, wi);
    float dist = sqrtf(dist2);
    if (dist < kRayEps) {
        return false;
    }
    wi = mul3(wi, 1.0f / dist);
    float cosO = dot3(N, wi);
    if (cosO <= 0.0f) {
        return false;
    }
    float3 lightN = norm3(cross3(sub3(load3(area.v1), load3(area.v0)), sub3(load3(area.v2), load3(area.v0))));
    float cosL = dot3(lightN, mul3(wi, -1.0f));
    if (cosL <= 0.0f) {
        return false;
    }
    if (isSegmentOccluded(scene, hitPoint, lightPoint, N)) {
        return false;
    }
    float lightArea = triangleArea(load3(area.v0), load3(area.v1), load3(area.v2));
    float3 emission = load3(area.color);
    if (useMis) {
        float pdfLight = dist2 / (lightArea * cosL);
        float pdfBrdf = pdfDiffuse(cosO);
        float misW = misWeightPower(pdfLight, pdfLight, pdfBrdf);
        if (misW < 1e-8f) {
            return false;
        }
        contrib = add3(contrib,
                        mul3(mul3v(albedo, emission), cosO / (kPi * pdfLight) * misW));
    } else {
        contrib = add3(contrib, mul3(mul3v(albedo, emission), cosO * cosL * lightArea / (kPi * dist2)));
    }
    return true;
}

__device__ float3 sampleDirectEmissive(const GpuSceneDevice &scene, float3 hitPoint, float3 N,
                                       float3 albedo, bool useMis, curandState &rng) {
    float3 direct = make3(0, 0, 0);
    for (int i = 0; i < scene.numAreaLights; ++i) {
        sampleAreaLightDiffuse(scene, scene.areaLights[i], hitPoint, N, albedo, useMis, direct, rng);
    }
    return direct;
}

__device__ bool sampleAreaLightGlossy(const GpuSceneDevice &scene, const GpuAreaLight &area,
                                      float3 hitPoint, float3 N, float3 wo, const GpuMaterial &mat,
                                      bool useMis, float3 &contrib, curandState &rng) {
    float3 lightPoint = sampleTriangle(load3(area.v0), load3(area.v1), load3(area.v2), rng);
    float3 wi = sub3(lightPoint, hitPoint);
    float dist2 = dot3(wi, wi);
    float dist = sqrtf(dist2);
    if (dist < kRayEps) {
        return false;
    }
    wi = mul3(wi, 1.0f / dist);
    float cosO = dot3(N, wi);
    if (cosO <= 0.0f) {
        return false;
    }
    float3 lightN = norm3(cross3(sub3(load3(area.v1), load3(area.v0)), sub3(load3(area.v2), load3(area.v0))));
    float cosL = dot3(lightN, mul3(wi, -1.0f));
    if (cosL <= 0.0f) {
        return false;
    }
    if (isSegmentOccluded(scene, hitPoint, lightPoint, N)) {
        return false;
    }
    float lightArea = triangleArea(load3(area.v0), load3(area.v1), load3(area.v2));
    float3 brdf = evalGlossy(N, wo, wi, mat);
    float3 emission = load3(area.color);
    if (useMis) {
        float pdfLight = dist2 / (lightArea * cosL);
        float pdfBrdf = pdfGlossy(N, wo, wi, mat);
        float misW = misWeightPower(pdfLight, pdfLight, pdfBrdf);
        if (misW < 1e-8f) {
            return false;
        }
        contrib = add3(contrib, mul3(mul3v(brdf, emission), cosO / pdfLight * misW));
    } else {
        contrib = add3(contrib, mul3(mul3v(brdf, emission), cosO * cosL * lightArea / dist2));
    }
    return true;
}

__device__ float3 sampleDirectEmissiveBRDF(const GpuSceneDevice &scene, float3 hitPoint, float3 N,
                                           float3 wo, const GpuMaterial &mat, bool useMis,
                                           curandState &rng) {
    float3 direct = make3(0, 0, 0);
    for (int i = 0; i < scene.numAreaLights; ++i) {
        sampleAreaLightGlossy(scene, scene.areaLights[i], hitPoint, N, wo, mat, useMis, direct, rng);
    }
    return direct;
}

__device__ float3 sampleDirectPointLights(const GpuSceneDevice &scene, float3 hitPoint, float3 N,
                                          float3 albedo) {
    float3 direct = make3(0, 0, 0);
    if (scene.pointLights == nullptr || scene.numPointLights <= 0) {
        return direct;
    }
    for (int i = 0; i < scene.numPointLights; ++i) {
        const GpuPointLight &pl = scene.pointLights[i];
        float3 toLight = sub3(load3(pl.pos), hitPoint);
        float dist2 = dot3(toLight, toLight);
        if (dist2 < kRayEps * kRayEps) {
            continue;
        }
        float3 L = mul3(toLight, 1.0f / sqrtf(dist2));
        if (dot3(N, L) <= 0.0f) {
            continue;
        }
        if (isSegmentOccluded(scene, hitPoint, load3(pl.pos), N)) {
            continue;
        }
        float cosTheta = dot3(N, L);
        direct = add3(direct, mul3(mul3v(albedo, load3(pl.color)), cosTheta / dist2));
    }
    return direct;
}

__device__ float3 sampleDirectPointLightsGlossy(const GpuSceneDevice &scene, float3 hitPoint,
                                                float3 N, float3 wo, const GpuMaterial &mat) {
    float3 direct = make3(0, 0, 0);
    if (scene.pointLights == nullptr || scene.numPointLights <= 0) {
        return direct;
    }
    for (int i = 0; i < scene.numPointLights; ++i) {
        const GpuPointLight &pl = scene.pointLights[i];
        float3 toLight = sub3(load3(pl.pos), hitPoint);
        float dist2 = dot3(toLight, toLight);
        if (dist2 < kRayEps * kRayEps) {
            continue;
        }
        float3 L = mul3(toLight, 1.0f / sqrtf(dist2));
        if (dot3(N, L) <= 0.0f) {
            continue;
        }
        if (isSegmentOccluded(scene, hitPoint, load3(pl.pos), N)) {
            continue;
        }
        float cosTheta = dot3(N, L);
        float3 brdf = evalGlossy(N, wo, L, mat);
        direct = add3(direct, mul3(mul3v(brdf, load3(pl.color)), cosTheta / dist2));
    }
    return direct;
}

__device__ float survivalProb(float3 throughput) {
    return fmaxf(kRrMinSurvival, luminance3(throughput));
}

struct MisCtx {
    float pdfBrdf;
    float3 wi;
    float3 shadingPoint;
    float3 N;
    bool active;
    bool glossyPath;
};

__device__ float3 castRayPath(const GpuSceneDevice &scene, float3 orig, float3 dir, int depth,
                              float3 throughput, bool countEmissive, int mode,
                              bool dispersionEnabled, int dispChannel, const MisCtx *misCtx,
                              const GpuGuidingGrid *guideGrid, bool trainingPass,
                              curandState &rng);

__device__ float3 castRayPath(const GpuSceneDevice &scene, float3 orig, float3 dir, int depth,
                              float3 throughput, bool countEmissive, int mode,
                              bool dispersionEnabled, int dispChannel, const MisCtx *misCtx,
                              const GpuGuidingGrid *guideGrid, bool trainingPass,
                              curandState &rng) {
    if (depth > kMaxDepth) {
        return make3(0, 0, 0);
    }

    HitRec hit;
    if (!intersectScene(scene, orig, dir, kRayEps, hit)) {
        return make3(0, 0, 0);
    }

    const GpuMaterial &mat = scene.materials[hit.matId < 0 || hit.matId >= scene.numMaterials ? 0 : hit.matId];
    float3 hitPoint = add3(orig, mul3(dir, hit.t));
    float3 D = norm3(dir);
    float3 geomN = norm3(load3(hit.n));

    if (mat.type == GPU_MAT_EMISSIVE) {
        if (!countEmissive) {
            return make3(0, 0, 0);
        }
        float3 emission = load3(mat.emission);
        if (misCtx != nullptr && misCtx->active) {
            float pdfLight = computeAreaLightPdf(scene, misCtx->shadingPoint, misCtx->wi);
            float misDenom = misPowerDenom(pdfLight, misCtx->pdfBrdf);
            if (misDenom < 1e-8f) {
                return make3(0, 0, 0);
            }
            float misW = misWeightPower(misCtx->pdfBrdf, misCtx->pdfBrdf, pdfLight);
            float scale = misW / misCtx->pdfBrdf;
            return clampRadiance3(mul3(mul3v(throughput, emission), scale));
        }
        return clampRadiance3(mul3v(throughput, emission));
    }

    bool opaqueBack = mat.type != GPU_MAT_REFRACT && mat.type != GPU_MAT_EMISSIVE && dot3(D, geomN) > 0.0f;
    if (opaqueBack) {
        return make3(0, 0, 0);
    }

    bool useNee = mode >= GPU_PATH_NEE;
    bool useMis = mode == GPU_PATH_MIS;
    bool useGuide = guideGrid != nullptr && (useGuidingMode(mode) || trainingPass);

    if (mat.type == GPU_MAT_REFLECT) {
        float3 N = faceNormal(D, geomN);
        float3 reflected = sub3(D, mul3(N, 2.0f * dot3(D, N)));
        float3 origin = offsetAlongNormal(hitPoint, N, kOriginOffset);
        float3 newTp = scaleDispAttenuation(throughput, load3(mat.specular), dispChannel);
        return clampRadiance3(castRayPath(scene, origin, reflected, depth + 1, newTp, true, mode,
                                          dispersionEnabled, dispChannel, nullptr, guideGrid,
                                          trainingPass, rng));
    }

    if (mat.type == GPU_MAT_REFRACT) {
        float3 refractColor = load3(mat.specular);
        bool exiting = dot3(D, geomN) > 0.0f;
        // Exit split only (entry stays single-ray): different IOR per channel projects rainbow on screen.
        if (dispersionEnabled && mat.dispersionDelta > 0.0f && dispChannel < 0 && exiting) {
            float rx = 0.0f;
            float gy = 0.0f;
            float bz = 0.0f;
            for (int c = 0; c < 3; ++c) {
                float ior = channelIor(mat.ior, mat.dispersionDelta, c);
                float3 newDir = computeRefractDirection(D, geomN, ior);
                float3 origin = offsetAlongRay(hitPoint, newDir, kRefractOriginOffset);
                float3 chTp = scaleDispAttenuation(throughput, refractColor, c);
                float3 child = castRayPath(scene, origin, newDir, depth + 1, chTp, true, mode,
                                           dispersionEnabled, c, nullptr, guideGrid, trainingPass,
                                           rng);
                if (c == 0) {
                    rx = child.x;
                } else if (c == 1) {
                    gy = child.y;
                } else {
                    bz = child.z;
                }
            }
            return clampRadiance3(make3(rx, gy, bz));
        }
        float ior = mat.ior;
        if (dispChannel >= 0 && mat.dispersionDelta > 0.0f) {
            ior = channelIor(mat.ior, mat.dispersionDelta, dispChannel);
        }
        float3 newDir = computeRefractDirection(D, geomN, ior);
        float3 origin = offsetAlongRay(hitPoint, newDir, kRefractOriginOffset);
        float3 newTp = scaleDispAttenuation(throughput, refractColor, dispChannel);
        return clampRadiance3(castRayPath(scene, origin, newDir, depth + 1, newTp, true, mode,
                                          dispersionEnabled, dispChannel, nullptr, guideGrid,
                                          trainingPass, rng));
    }

    if (mat.type == GPU_MAT_GLOSSY) {
        float3 wo = mul3(D, -1.0f);
        float3 N = shadingNormal(wo, geomN);
        float3 kd = load3(mat.diffuse);
        float3 ks = load3(mat.specular);
        float3 direct = make3(0, 0, 0);
        float3 indirect = make3(0, 0, 0);
        if (useNee) {
            direct = sampleDirectEmissiveBRDF(scene, hitPoint, N, wo, mat, useMis, rng);
            direct = add3(direct, sampleDirectPointLightsGlossy(scene, hitPoint, N, wo, mat));
        }

        float kdLum = luminance3(kd);
        float ksLum = luminance3(ks);
        bool isMetal = kdLum < 0.01f;
        float specProb = isMetal ? 1.0f : ksLum / fmaxf(1e-4f, kdLum + ksLum);
        float3 wi;
        float pdf = 0.0f;
        float3 brdf = make3(0, 0, 0);
        bool specularLobe = isMetal || gpuUniform(rng) < specProb;
        if (trainingPass && useGuide) {
            guideDeposit(*guideGrid, hitPoint, N, mul3(D, -1.0f), luminance3(throughput));
        }
        if (specularLobe) {
            float u1 = gpuUniform(rng);
            float u2 = gpuUniform(rng);
            float phi = 2.0f * kPi * u1;
            float m2 = mat.roughness * mat.roughness;
            float tanTheta2 = -m2 * logf(fmaxf(1e-6f, 1.0f - u2));
            float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float3 tangent, bitangent;
            buildBasis(N, tangent, bitangent);
            float3 h = norm3(add3(add3(mul3(tangent, cosf(phi) * sinTheta),
                                        mul3(bitangent, sinf(phi) * sinTheta)),
                                    mul3(N, cosTheta)));
            if (dot3(h, wo) < 0.0f) {
                h = mul3(h, -1.0f);
            }
            wi = norm3(sub3(mul3(h, 2.0f * dot3(wo, h)), wo));
            if (dot3(N, wi) <= 0.0f) {
                return direct;
            }
            brdf = evalSpecular(N, wo, wi, ks, mat.roughness, load3(mat.f0));
            pdf = pdfGlossy(N, wo, wi, mat);
        } else if (useGuide && !trainingPass) {
            float pdfTotal = 0.0f;
            if (sampleIndirectWithGuide(*guideGrid, hitPoint, N, wi, pdfTotal, rng)) {
                float cosO = fmaxf(0.0f, dot3(N, wi));
                float3 brdfVal = mul3(kd, 1.0f / kPi);
                float rrProb = 1.0f;
                bool traceIndirect = true;
                if (depth >= kRrStartDepth) {
                    rrProb = survivalProb(mul3v(throughput, add3(kd, ks)));
                    traceIndirect = gpuUniform(rng) <= rrProb;
                }
                if (traceIndirect) {
                    float3 origin = offsetAlongNormal(hitPoint, N, kOriginOffset);
                    bool indirectEmissive = !useNee || useMis;
                    float3 Li = castRayPath(scene, origin, wi, depth + 1, throughput, indirectEmissive,
                                            mode, dispersionEnabled, dispChannel, nullptr, guideGrid,
                                            trainingPass, rng);
                    indirect = clampRadiance3(mul3(mul3v(brdfVal, Li), cosO / (pdfTotal * rrProb)));
                }
                return add3(direct, clampRadiance3(indirect));
            }
        } else {
            wi = sampleCosineHemisphere(N, pdf, rng);
            brdf = mul3(kd, 1.0f / kPi);
            pdf = pdfGlossy(N, wo, wi, mat);
        }

        if (pdf >= 1e-8f) {
            float rrProb = 1.0f;
            bool traceIndirect = true;
            if (depth >= kRrStartDepth) {
                rrProb = survivalProb(mul3v(throughput, add3(kd, ks)));
                traceIndirect = gpuUniform(rng) <= rrProb;
            }
            if (traceIndirect) {
                float cosO = fmaxf(0.0f, dot3(N, wi));
                float3 origin = offsetAlongNormal(hitPoint, N, kOriginOffset);
                bool indirectEmissive = !useNee || useMis;
                MisCtx mis;
                const MisCtx *misPtr = nullptr;
                if (useMis) {
                    mis.pdfBrdf = pdf;
                    mis.wi = wi;
                    mis.shadingPoint = hitPoint;
                    mis.N = N;
                    mis.active = true;
                    mis.glossyPath = true;
                    misPtr = &mis;
                }
                float3 Li = castRayPath(scene, origin, wi, depth + 1, throughput, indirectEmissive,
                                        mode, dispersionEnabled, dispChannel, misPtr, guideGrid,
                                        trainingPass, rng);
                indirect = clampRadiance3(mul3(mul3v(brdf, Li), cosO / (pdf * rrProb)));
            }
        }
        return add3(direct, clampRadiance3(indirect));
    }

    float3 N = shadingNormal(mul3(D, -1.0f), geomN);
    float3 albedo = load3(mat.diffuse);
    float3 direct = make3(0, 0, 0);
    float3 indirect = make3(0, 0, 0);
    if (useNee) {
        direct = add3(sampleDirectEmissive(scene, hitPoint, N, albedo, useMis, rng),
                      sampleDirectPointLights(scene, hitPoint, N, albedo));
    }

    if (trainingPass && useGuide) {
        guideDeposit(*guideGrid, hitPoint, N, mul3(D, -1.0f), luminance3(throughput));
    }

    float3 wi;
    float pdf = 0.0f;
    if (useGuide && !trainingPass) {
        float pdfTotal = 0.0f;
        if (sampleIndirectWithGuide(*guideGrid, hitPoint, N, wi, pdfTotal, rng)) {
            float cosO = fmaxf(0.0f, dot3(N, wi));
            float3 brdfVal = mul3(albedo, 1.0f / kPi);
            float rrProb = 1.0f;
            bool traceIndirect = true;
            if (depth >= kRrStartDepth) {
                rrProb = survivalProb(mul3v(throughput, albedo));
                traceIndirect = gpuUniform(rng) <= rrProb;
            }
            if (traceIndirect) {
                float3 origin = offsetAlongNormal(hitPoint, N, kOriginOffset);
                bool indirectEmissive = !useNee || useMis;
                float3 Li = castRayPath(scene, origin, wi, depth + 1, throughput, indirectEmissive, mode,
                                        dispersionEnabled, dispChannel, nullptr, guideGrid,
                                        trainingPass, rng);
                indirect = clampRadiance3(mul3(mul3v(brdfVal, Li), cosO / (pdfTotal * rrProb)));
            }
            return add3(direct, clampRadiance3(indirect));
        }
    }

    wi = sampleCosineHemisphere(N, pdf, rng);
    if (pdf >= 1e-8f) {
        float rrProb = 1.0f;
        bool traceIndirect = true;
        if (depth >= kRrStartDepth) {
            rrProb = survivalProb(mul3v(throughput, albedo));
            traceIndirect = gpuUniform(rng) <= rrProb;
        }
        if (traceIndirect) {
            float3 origin = offsetAlongNormal(hitPoint, N, kOriginOffset);
            bool indirectEmissive = !useNee || useMis;
            MisCtx mis;
            const MisCtx *misPtr = nullptr;
            if (useMis) {
                mis.pdfBrdf = pdf;
                mis.wi = wi;
                mis.shadingPoint = hitPoint;
                mis.N = N;
                mis.active = true;
                mis.glossyPath = false;
                misPtr = &mis;
            }
            float3 Li = castRayPath(scene, origin, wi, depth + 1, throughput, indirectEmissive, mode,
                                    dispersionEnabled, dispChannel, misPtr, guideGrid, trainingPass,
                                    rng);
            indirect = clampRadiance3(mul3(mul3v(albedo, Li), 1.0f / rrProb));
        }
    }
    return add3(direct, clampRadiance3(indirect));
}

__device__ bool isInShadow(const GpuSceneDevice &scene, float3 p, float3 N, float3 L, float maxT);

__device__ float3 shadePhongWhitted(const GpuMaterial &mat, float3 N, float3 V, float3 L,
                                    float3 lightColor, float attenuation) {
    float d = dot3(N, L);
    if (d <= 0.0f) {
        return make3(0, 0, 0);
    }
    float3 kd = load3(mat.diffuse);
    float3 ks = load3(mat.specular);
    float3 diffuse = mul3(mul3v(kd, lightColor), d * attenuation);
    float3 R = sub3(mul3(N, 2.0f * d), L);
    float spec = dot3(R, V);
    if (spec <= 0.0f || mat.shininess <= 0.0f) {
        return diffuse;
    }
    float s = powf(spec, mat.shininess);
    return add3(diffuse, mul3(mul3v(ks, lightColor), s * attenuation));
}

__device__ float3 shadeWhittedLights(const GpuSceneDevice &scene, float3 hitPoint, float3 geomN,
                                     float3 rayDir, const GpuMaterial &mat) {
    float3 shadowN = shadingNormal(rayDir, geomN);
    float3 viewToCam = mul3(rayDir, -1.0f);
    float3 shadeN = shadingNormal(viewToCam, geomN);
    float3 color = make3(0, 0, 0);
    for (int i = 0; i < scene.numPointLights; ++i) {
        const GpuPointLight &pl = scene.pointLights[i];
        float3 toLight = sub3(load3(pl.pos), hitPoint);
        float dist2 = dot3(toLight, toLight);
        if (dist2 < kRayEps * kRayEps) {
            continue;
        }
        float dist = sqrtf(dist2);
        float3 L = mul3(toLight, 1.0f / dist);
        if (isInShadow(scene, hitPoint, shadowN, L, dist)) {
            continue;
        }
        color = add3(color, shadePhongWhitted(mat, shadeN, viewToCam, L, load3(pl.color), 1.0f));
    }
    for (int i = 0; i < scene.numDirectionalLights; ++i) {
        const GpuDirectionalLight &dl = scene.directionalLights[i];
        float3 L = load3(dl.direction);
        if (isInShadow(scene, hitPoint, shadowN, L, 1e30f)) {
            continue;
        }
        color = add3(color, shadePhongWhitted(mat, shadeN, viewToCam, L, load3(dl.color), 1.0f));
    }
    return color;
}

__device__ bool isInShadow(const GpuSceneDevice &scene, float3 p, float3 N, float3 L, float maxT) {
    float3 shadowN = N;
    if (dot3(shadowN, L) < 0.0f) {
        shadowN = mul3(shadowN, -1.0f);
    }
    float3 shadowOrigin = add3(p, mul3(shadowN, kShadowEps));
    HitRec hit;
    if (!intersectScene(scene, shadowOrigin, L, kRayEps, hit)) {
        return false;
    }
    if (hit.t >= maxT - kShadowEps) {
        return false;
    }
    if (scene.materials[hit.matId < 0 || hit.matId >= scene.numMaterials ? 0 : hit.matId].type == GPU_MAT_REFRACT) {
        return false;
    }
    return true;
}

__device__ float3 castRayWhitted(const GpuSceneDevice &scene, float3 orig, float3 dir, int depth,
                                 float tmin, bool dispersionEnabled, curandState &rng) {
    (void)rng;
    if (depth > kMaxDepth) {
        return load3(scene.camera.bg);
    }
    HitRec hit;
    if (!intersectScene(scene, orig, dir, tmin, hit)) {
        return load3(scene.camera.bg);
    }
    const GpuMaterial &mat = scene.materials[hit.matId < 0 || hit.matId >= scene.numMaterials ? 0 : hit.matId];
    float3 hitPoint = add3(orig, mul3(dir, hit.t));
    float3 D = norm3(dir);
    float3 geomN = norm3(load3(hit.n));

    if (mat.type == GPU_MAT_EMISSIVE) {
        return load3(mat.emission);
    }
    if (mat.type != GPU_MAT_REFRACT && mat.type != GPU_MAT_EMISSIVE && dot3(D, geomN) > 0.0f) {
        return make3(0, 0, 0);
    }

    if (mat.type == GPU_MAT_DIFFUSE || mat.type == GPU_MAT_GLOSSY) {
        return shadeWhittedLights(scene, hitPoint, geomN, D, mat);
    }

    if (mat.type == GPU_MAT_REFLECT) {
        float3 N = faceNormal(D, geomN);
        float3 reflected = sub3(D, mul3(N, 2.0f * dot3(D, N)));
        float3 origin = offsetAlongNormal(hitPoint, N, kOriginOffset);
        float3 child = castRayWhitted(scene, origin, reflected, depth + 1, kRayEps, dispersionEnabled, rng);
        return mul3v(child, load3(mat.specular));
    }

    float3 refractColor = load3(mat.specular);
    bool exiting = dot3(D, geomN) > 0.0f;
    if (dispersionEnabled && mat.dispersionDelta > 0.0f && exiting) {
        float rx = 0.0f;
        float gy = 0.0f;
        float bz = 0.0f;
        for (int c = 0; c < 3; ++c) {
            float ior = channelIor(mat.ior, mat.dispersionDelta, c);
            float3 newDir = computeRefractDirection(D, geomN, ior);
            float3 origin = offsetAlongRay(hitPoint, newDir, kRefractOriginOffset);
            float3 child = castRayWhitted(scene, origin, newDir, depth + 1, kRefractRayTMin,
                                          dispersionEnabled, rng);
            if (c == 0) {
                rx = child.x * refractColor.x;
            } else if (c == 1) {
                gy = child.y * refractColor.y;
            } else {
                bz = child.z * refractColor.z;
            }
        }
        return make3(rx, gy, bz);
    }

    float3 newDir = computeRefractDirection(D, geomN, mat.ior);
    float3 origin = offsetAlongRay(hitPoint, newDir, kRefractOriginOffset);
    float3 child = castRayWhitted(scene, origin, newDir, depth + 1, kRefractRayTMin, dispersionEnabled, rng);
    return mul3v(child, refractColor);
}

__device__ float3 generateCameraRay(const GpuCamera &cam, float px, float py) {
    float3 Rc = make3((px - cam.cx) / cam.fx, (cam.cy - py) / cam.fy, 1.0f);
    float3 horiz = load3(cam.horizontal);
    float3 up = load3(cam.up);
    float3 dir = load3(cam.direction);
    float3 R_w = add3(add3(mul3(horiz, Rc.x), mul3(up, -Rc.y)), mul3(dir, Rc.z));
    return load3(cam.center);
}

__device__ float3 generateCameraDir(const GpuCamera &cam, float px, float py) {
    float3 Rc = make3((px - cam.cx) / cam.fx, (cam.cy - py) / cam.fy, 1.0f);
    float3 horiz = load3(cam.horizontal);
    float3 up = load3(cam.up);
    float3 dir = load3(cam.direction);
    return norm3(add3(add3(mul3(horiz, Rc.x), mul3(up, -Rc.y)), mul3(dir, Rc.z)));
}

__device__ float hash01(int a, int b, int c) {
    unsigned int x = static_cast<unsigned int>(a * 374761393 + b * 668265263 + c * 1274126177);
    x = (x ^ (x >> 13)) * 1274126177u;
    x ^= x >> 16;
    return (x & 0xFFFFFFu) / float(0x1000000u);
}

__global__ void initCurandKernel(curandState *states, int width, int height, unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int idx = y * width + x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void renderKernel(const GpuSceneDevice scene, float *output, curandState *rngStates,
                             int width, int height, int spp, int mode, bool dispersion,
                             const GpuGuidingGrid guideGrid, bool useGuideGrid) {
    // One thread ↔ one pixel; SPP is serial inside the thread (same structure as CPU main loop).
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    curandState localState = rngStates[y * width + x];
    float3 accum = make3(0, 0, 0);
    const GpuGuidingGrid *guidePtr = useGuideGrid ? &guideGrid : nullptr;
    for (int s = 0; s < spp; ++s) {
        float jx = float(x);
        float jy = float(y);
        if (spp > 1) {
            jx += hash01(x, y, s);
            jy += hash01(y, x, s + 31);
        }
        float3 orig = load3(scene.camera.center);
        float3 dir = generateCameraDir(scene.camera, jx, jy);
        float3 sampleColor;
        if (mode == GPU_WHITTED) {
            sampleColor = castRayWhitted(scene, orig, dir, 0, kRayEps, dispersion, localState);
        } else {
            bool countEmissive = !(mode == GPU_PATH_NEE || mode == GPU_PATH_GUIDING);
            sampleColor = castRayPath(scene, orig, dir, 0, make3(1, 1, 1), countEmissive, mode,
                                      dispersion, -1, nullptr, guidePtr, false, localState);
        }
        accum = add3(accum, sampleColor);
    }
    rngStates[y * width + x] = localState;
    accum = mul3(accum, 1.0f / spp);
    int idx = (y * width + x) * 3;
    output[idx + 0] = accum.x;
    output[idx + 1] = accum.y;
    output[idx + 2] = accum.z;
}

__global__ void trainGuideKernel(const GpuSceneDevice scene, curandState *rngStates, int width,
                                 int height, int trainSpp, GpuGuidingGrid guideGrid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    curandState localState = rngStates[y * width + x];
    for (int s = 0; s < trainSpp; ++s) {
        float jx = float(x);
        float jy = float(y);
        if (trainSpp > 1) {
            jx += hash01(x, y, s + 101);
            jy += hash01(y, x, s + 211);
        }
        float3 orig = load3(scene.camera.center);
        float3 dir = generateCameraDir(scene.camera, jx, jy);
        (void)castRayPath(scene, orig, dir, 0, make3(1, 1, 1), true, GPU_PATH_GUIDING, false, -1,
                          nullptr, &guideGrid, true, localState);
    }
    rngStates[y * width + x] = localState;
}

__global__ void normalizeGuideKernel(GpuGuidingGrid grid) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= kGuideNumCells) {
        return;
    }
    float sum = 0.0f;
    int base = cell * grid.binsPerCell;
    for (int b = 0; b < grid.binsPerCell; ++b) {
        sum += grid.weights[base + b];
    }
    if (sum < 1e-8f) {
        // Leave empty cells at zero — fall back to cosine BRDF sampling there.
        return;
    }
    float inv = 1.0f / sum;
    for (int b = 0; b < grid.binsPerCell; ++b) {
        grid.weights[base + b] *= inv;
    }
}

struct DeviceBuffers {
    GpuMaterial *materials = nullptr;
    GpuSphere *spheres = nullptr;
    GpuPlane *planes = nullptr;
    GpuTriangle *triangles = nullptr;
    GpuAreaLight *areaLights = nullptr;
    GpuPointLight *pointLights = nullptr;
    GpuDirectionalLight *directionalLights = nullptr;
    GpuSceneDevice scene{};
    float *pixels = nullptr;
    curandState *rngStates = nullptr;
    float *guideWeights = nullptr;
    GpuGuidingGrid guideGrid{};
    int cachedWidth = 0;
    int cachedHeight = 0;
    size_t sceneHash = 0;
};

static DeviceBuffers g_device;

static bool initGuideGrid(const GpuSceneHost &host) {
    size_t numFloats = static_cast<size_t>(kGuideNumCells) * kGuideBinsPerCell;
    if (cudaMalloc(&g_device.guideWeights, sizeof(float) * numFloats) != cudaSuccess) {
        return false;
    }
    if (cudaMemset(g_device.guideWeights, 0, sizeof(float) * numFloats) != cudaSuccess) {
        return false;
    }
    g_device.guideGrid.res = kGuideGridRes;
    g_device.guideGrid.thetaBins = kGuideThetaBins;
    g_device.guideGrid.phiBins = kGuidePhiBins;
    g_device.guideGrid.binsPerCell = kGuideBinsPerCell;
    g_device.guideGrid.weights = g_device.guideWeights;
    if (host.hasBbox) {
        g_device.guideGrid.bboxMin[0] = host.bboxMin[0];
        g_device.guideGrid.bboxMin[1] = host.bboxMin[1];
        g_device.guideGrid.bboxMin[2] = host.bboxMin[2];
        g_device.guideGrid.bboxMax[0] = host.bboxMax[0];
        g_device.guideGrid.bboxMax[1] = host.bboxMax[1];
        g_device.guideGrid.bboxMax[2] = host.bboxMax[2];
    } else {
        g_device.guideGrid.bboxMin[0] = -2.0f;
        g_device.guideGrid.bboxMin[1] = -2.0f;
        g_device.guideGrid.bboxMin[2] = -2.0f;
        g_device.guideGrid.bboxMax[0] = 2.0f;
        g_device.guideGrid.bboxMax[1] = 2.0f;
        g_device.guideGrid.bboxMax[2] = 2.0f;
    }
    return true;
}

static size_t hashScene(const GpuSceneHost &host) {
    size_t h = host.numMaterials ^ (host.numSpheres << 4) ^ (host.numPlanes << 8) ^
               (host.numTriangles << 12);
    return h;
}

static void freeDeviceBuffers() {
    if (g_device.materials) cudaFree(g_device.materials);
    if (g_device.spheres) cudaFree(g_device.spheres);
    if (g_device.planes) cudaFree(g_device.planes);
    if (g_device.triangles) cudaFree(g_device.triangles);
    if (g_device.areaLights) cudaFree(g_device.areaLights);
    if (g_device.pointLights) cudaFree(g_device.pointLights);
    if (g_device.directionalLights) cudaFree(g_device.directionalLights);
    if (g_device.pixels) cudaFree(g_device.pixels);
    if (g_device.rngStates) cudaFree(g_device.rngStates);
    if (g_device.guideWeights) cudaFree(g_device.guideWeights);
    g_device = DeviceBuffers{};
}

static bool uploadScene(const GpuSceneHost &host, int width, int height) {
    (void)width;
    (void)height;
    freeDeviceBuffers();
    auto check = [](cudaError_t e, const char *msg) -> bool {
        if (e != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(e));
            return false;
        }
        return true;
    };

    if (!check(cudaMalloc(&g_device.materials, sizeof(GpuMaterial) * host.numMaterials), "materials")) {
        return false;
    }
    if (!check(cudaMemcpy(g_device.materials, host.materials, sizeof(GpuMaterial) * host.numMaterials,
                          cudaMemcpyHostToDevice), "copy materials")) {
        return false;
    }

    if (host.numSpheres > 0) {
        if (!check(cudaMalloc(&g_device.spheres, sizeof(GpuSphere) * host.numSpheres), "spheres")) {
            return false;
        }
        if (!check(cudaMemcpy(g_device.spheres, host.spheres, sizeof(GpuSphere) * host.numSpheres,
                              cudaMemcpyHostToDevice), "copy spheres")) {
            return false;
        }
    }
    if (host.numPlanes > 0) {
        if (!check(cudaMalloc(&g_device.planes, sizeof(GpuPlane) * host.numPlanes), "planes")) {
            return false;
        }
        if (!check(cudaMemcpy(g_device.planes, host.planes, sizeof(GpuPlane) * host.numPlanes,
                              cudaMemcpyHostToDevice), "copy planes")) {
            return false;
        }
    }
    if (host.numTriangles > 0) {
        if (!check(cudaMalloc(&g_device.triangles, sizeof(GpuTriangle) * host.numTriangles), "tris")) {
            return false;
        }
        if (!check(cudaMemcpy(g_device.triangles, host.triangles, sizeof(GpuTriangle) * host.numTriangles,
                              cudaMemcpyHostToDevice), "copy tris")) {
            return false;
        }
    }
    if (host.numAreaLights > 0) {
        if (!check(cudaMalloc(&g_device.areaLights, sizeof(GpuAreaLight) * host.numAreaLights), "area")) {
            return false;
        }
        if (!check(cudaMemcpy(g_device.areaLights, host.areaLights, sizeof(GpuAreaLight) * host.numAreaLights,
                              cudaMemcpyHostToDevice), "copy area")) {
            return false;
        }
    }
    if (host.numPointLights > 0) {
        if (!check(cudaMalloc(&g_device.pointLights, sizeof(GpuPointLight) * host.numPointLights), "point")) {
            return false;
        }
        if (!check(cudaMemcpy(g_device.pointLights, host.pointLights,
                              sizeof(GpuPointLight) * host.numPointLights,
                              cudaMemcpyHostToDevice), "copy point")) {
            return false;
        }
    }
    if (host.numDirectionalLights > 0) {
        if (!check(cudaMalloc(&g_device.directionalLights,
                              sizeof(GpuDirectionalLight) * host.numDirectionalLights),
                   "directional")) {
            return false;
        }
        if (!check(cudaMemcpy(g_device.directionalLights, host.directionalLights,
                              sizeof(GpuDirectionalLight) * host.numDirectionalLights,
                              cudaMemcpyHostToDevice), "copy directional")) {
            return false;
        }
    }

    if (!check(cudaMalloc(&g_device.pixels, sizeof(float) * 3 * width * height), "pixels")) {
        return false;
    }
    if (!check(cudaMalloc(&g_device.rngStates, sizeof(curandState) * width * height), "rng")) {
        return false;
    }

    if (!initGuideGrid(host)) {
        return false;
    }

    g_device.scene.materials = g_device.materials;
    g_device.scene.numMaterials = host.numMaterials;
    g_device.scene.spheres = g_device.spheres;
    g_device.scene.numSpheres = host.numSpheres;
    g_device.scene.planes = g_device.planes;
    g_device.scene.numPlanes = host.numPlanes;
    g_device.scene.triangles = g_device.triangles;
    g_device.scene.numTriangles = host.numTriangles;
    g_device.scene.areaLights = g_device.areaLights;
    g_device.scene.numAreaLights = host.numAreaLights;
    g_device.scene.pointLights = g_device.pointLights;
    g_device.scene.numPointLights = host.numPointLights;
    g_device.scene.directionalLights = g_device.directionalLights;
    g_device.scene.numDirectionalLights = host.numDirectionalLights;
    g_device.scene.camera = host.camera;
    g_device.sceneHash = hashScene(host);
    g_device.cachedWidth = width;
    g_device.cachedHeight = height;
    return true;
}

} // namespace

bool cudaAvailable() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        return false;
    }
    return true;
}

void freeCudaSceneCache() {
    freeDeviceBuffers();
}

bool renderWithCuda(const SceneParser &scene, Image &image, RenderMode mode, int spp,
                    bool dispersion, double &renderSec, int trainSpp) {
    if (!cudaAvailable()) {
        return false;
    }

    GpuSceneHost host = buildGpuSceneHost(scene);
    int width = scene.getCamera()->getWidth();
    int height = scene.getCamera()->getHeight();

    fprintf(stderr, "CUDA scene: materials=%d spheres=%d planes=%d triangles=%d area=%d point=%d dir=%d\n",
            host.numMaterials, host.numSpheres, host.numPlanes, host.numTriangles,
            host.numAreaLights, host.numPointLights, host.numDirectionalLights);

    if (!uploadScene(host, width, height)) {
        return false;
    }

    if (host.numPointLights > 0) {
        GpuPointLight hpl{};
        if (cudaMemcpy(&hpl, g_device.pointLights, sizeof(GpuPointLight), cudaMemcpyDeviceToHost) == cudaSuccess) {
            fprintf(stderr, "CUDA point light: pos=(%.2f,%.2f,%.2f) col=(%.2f,%.2f,%.2f)\n",
                    hpl.pos[0], hpl.pos[1], hpl.pos[2], hpl.color[0], hpl.color[1], hpl.color[2]);
        }
    }

    cudaDeviceSetLimit(cudaLimitStackSize, 65536);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int gpuMode = GPU_WHITTED;
    if (mode == RenderMode::PATH_TRACE) {
        gpuMode = GPU_PATH;
    } else if (mode == RenderMode::PATH_TRACE_NEE) {
        gpuMode = GPU_PATH_NEE;
    } else if (mode == RenderMode::PATH_TRACE_MIS) {
        gpuMode = GPU_PATH_MIS;
    } else if (mode == RenderMode::PATH_TRACE_GUIDING) {
        gpuMode = GPU_PATH_GUIDING;
    }

    const bool useGuiding = mode == RenderMode::PATH_TRACE_GUIDING;
    int effectiveTrainSpp = trainSpp > 0 ? trainSpp : (spp < 128 ? 128 : spp);

    GpuSceneDevice sceneDev = g_device.scene;
    GpuGuidingGrid guideGridDev = g_device.guideGrid;
    auto t0 = std::chrono::high_resolution_clock::now();
    constexpr unsigned long long kCudaRenderSeed = 104729ULL;
    initCurandKernel<<<grid, block>>>(g_device.rngStates, width, height, kCudaRenderSeed);
    cudaError_t initErr = cudaGetLastError();
    if (initErr != cudaSuccess) {
        fprintf(stderr, "CUDA curand init error: %s\n", cudaGetErrorString(initErr));
        return false;
    }

    if (useGuiding) {
        fprintf(stderr, "CUDA path guiding: training pass (%d spp)...\n", effectiveTrainSpp);
        if (cudaMemset(g_device.guideWeights, 0,
                       sizeof(float) * kGuideNumCells * kGuideBinsPerCell) != cudaSuccess) {
            return false;
        }
        trainGuideKernel<<<grid, block>>>(sceneDev, g_device.rngStates, width, height,
                                          effectiveTrainSpp, guideGridDev);
        cudaError_t trainErr = cudaDeviceSynchronize();
        if (trainErr != cudaSuccess) {
            fprintf(stderr, "CUDA train kernel error: %s\n", cudaGetErrorString(trainErr));
            return false;
        }
        int normBlock = 256;
        int normGrid = (kGuideNumCells + normBlock - 1) / normBlock;
        normalizeGuideKernel<<<normGrid, normBlock>>>(guideGridDev);
        cudaError_t normErr = cudaDeviceSynchronize();
        if (normErr != cudaSuccess) {
            fprintf(stderr, "CUDA normalize guide error: %s\n", cudaGetErrorString(normErr));
            return false;
        }
        fprintf(stderr, "CUDA path guiding: render pass (%d spp)...\n", spp);
    }

    renderKernel<<<grid, block>>>(sceneDev, g_device.pixels, g_device.rngStates, width, height, spp,
                                  gpuMode, dispersion, guideGridDev, useGuiding);
    cudaError_t err = cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    renderSec = std::chrono::duration<double>(t1 - t0).count();

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return false;
    }

    std::vector<float> hostPixels(3 * width * height);
    if (cudaMemcpy(hostPixels.data(), g_device.pixels, sizeof(float) * hostPixels.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            image.SetPixel(x, y, Vector3f(hostPixels[idx], hostPixels[idx + 1], hostPixels[idx + 2]));
        }
    }
    return true;
}
