#ifndef RAYTRACER_H
#define RAYTRACER_H

// 文件说明：CPU 端 Whitted 与路径追踪（NEE/MIS/色散）核心。
// 原创性声明：参考已有代码（GAMES101、PBRT 概念与课件公式），
// 光路调度、NEE/MIS 权重与 Russian Roulette 等按作业需求独立实现。

#include <cmath>
#include <cfloat>
#include <algorithm>

#include "scene_parser.hpp"
#include "group.hpp"
#include "light.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "hit.hpp"

constexpr float RAY_EPSILON = 1e-4f;
constexpr float ORIGIN_OFFSET = 1e-3f;
constexpr float REFRACT_ORIGIN_OFFSET = 2e-3f;
constexpr float REFRACT_RAY_TMIN = 2e-3f;
constexpr float SHADOW_EPSILON = 1e-3f;
constexpr int MAX_TRACE_DEPTH = 12;
constexpr int RR_START_DEPTH = 8;
constexpr float RR_MIN_SURVIVAL = 0.15f;
constexpr float M_PI_F = 3.14159265358979323846f;
constexpr float PATH_RADIANCE_CLAMP = 100.0f;

enum class RenderMode {
    WHITTED,
    PATH_TRACE,
    PATH_TRACE_NEE,
    PATH_TRACE_MIS,
    PATH_TRACE_GUIDING
};

struct MisIndirectCtx {
    float pdfBrdf = 0.0f;
    Vector3f wi;
    Vector3f shadingPoint;
    Vector3f N;
    Vector3f wo;
    GlossyMaterial *glossyMat = nullptr;
};

namespace {
constexpr float AREA_LIGHT_VERT_EPS = 1e-4f;

bool areaLightVerticesEqual(const Vector3f &a, const Vector3f &b) {
    return (a - b).length() < AREA_LIGHT_VERT_EPS;
}

int countSharedAreaLightVertices(const AreaLight *a, const AreaLight *a2) {
    const Vector3f va[3] = {a->getVertex0(), a->getVertex1(), a->getVertex2()};
    const Vector3f vb[3] = {a2->getVertex0(), a2->getVertex1(), a2->getVertex2()};
    int shared = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (areaLightVerticesEqual(va[i], vb[j])) {
                ++shared;
                break;
            }
        }
    }
    return shared;
}

// Path-tracing scenes split emissive quads into two AreaLight triangles; Whitted Phong
// must not sum both or direct light is doubled (ceiling ghosting / blow-out).
bool isDuplicateSplitAreaLight(const SceneParser &scene, int lightIndex) {
    auto *al = dynamic_cast<AreaLight *>(scene.getLight(lightIndex));
    if (!al) {
        return false;
    }
    for (int j = 0; j < lightIndex; ++j) {
        if (auto *prev = dynamic_cast<AreaLight *>(scene.getLight(j))) {
            if (countSharedAreaLightVertices(al, prev) >= 2) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace

class RayTracer {
public:
    RayTracer(const SceneParser &scene, RenderMode mode = RenderMode::WHITTED, unsigned int seed = 0,
              bool dispersion = false) :
            scene(scene), mode(mode), rngState(seed ? seed : 1u), dispersionEnabled(dispersion) {
    }

    Vector3f trace(const Ray &ray) const {
        if (mode == RenderMode::WHITTED) {
            return castRayWhitted(ray, 0, RAY_EPSILON);
        }
        return castRayPath(ray, 0, Vector3f(1.0f, 1.0f, 1.0f), true);
    }

private:
    const SceneParser &scene;
    RenderMode mode;
    mutable unsigned int rngState;
    bool dispersionEnabled;

    bool useMIS() const {
        return mode == RenderMode::PATH_TRACE_MIS;
    }

    bool useNEE() const {
        return mode == RenderMode::PATH_TRACE_NEE || useMIS();
    }

    // Glossy area-light NEE uses power-heuristic MIS in path_nee as well: full BRDF
    // without MIS blows up at grazing view (small cos theta_r) toward bright emitters.
    bool useGlossyNEEMIS() const {
        return useNEE();
    }

    float uniform() const {
        rngState ^= rngState << 13;
        rngState ^= rngState >> 17;
        rngState ^= rngState << 5;
        return (rngState & 0x00FFFFFFu) / float(0x01000000u);
    }

    static float luminance(const Vector3f &c) {
        return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
    }

    static Vector3f clampRadiance(const Vector3f &c) {
        float lum = luminance(c);
        if (lum > PATH_RADIANCE_CLAMP) {
            return c * (PATH_RADIANCE_CLAMP / lum);
        }
        return c;
    }

    static Vector3f faceNormal(const Vector3f &dir, const Vector3f &geomN) {
        Vector3f n = geomN.normalized();
        if (Vector3f::dot(dir, n) > 0.0f) {
            n = -n;
        }
        return n;
    }

    static Vector3f offsetAlongRay(const Vector3f &p, const Vector3f &dir, float eps) {
        return p + dir.normalized() * eps;
    }

    static Vector3f offsetAlongNormal(const Vector3f &p, const Vector3f &n, float eps) {
        return p + n.normalized() * eps;
    }

    static bool isOpaqueBackFace(const Vector3f &dir, const Vector3f &geomN, MaterialType type) {
        return type != MaterialType::REFRACT && type != MaterialType::EMISSIVE &&
               Vector3f::dot(dir, geomN) > 0.0f;
    }

    // Snell refraction for a closed mesh with outward geometric normals.
    // dot(D, geomN) < 0: entering (air -> glass, eta = 1/ior)
    // dot(D, geomN) > 0: exiting  (glass -> air, eta = ior/1)
    static Vector3f computeRefractDirection(const Vector3f &D, const Vector3f &geomN, float ior) {
        float etai = 1.0f;
        float etat = ior;
        Vector3f n = geomN;
        float cosTheta = Vector3f::dot(D, n);
        if (cosTheta > 0.0f) {
            etai = ior;
            etat = 1.0f;
            n = -n;
            cosTheta = -cosTheta;
        }
        float eta = etai / etat;
        float k = 1.0f - eta * eta * (1.0f - cosTheta * cosTheta);
        if (k < 0.0f) {
            return (D - 2.0f * Vector3f::dot(D, n) * n).normalized();
        }
        return (eta * D + (eta * cosTheta - sqrtf(k)) * n).normalized();
    }

    struct RefractSplit {
        Vector3f refractDir;
        Vector3f reflectDir;
        bool tir = false;
    };

    static RefractSplit computeRefractSplit(const Vector3f &D, const Vector3f &geomN, float ior) {
        RefractSplit split{};
        float etai = 1.0f;
        float etat = ior;
        Vector3f n = geomN;
        float cosTheta = Vector3f::dot(D, n);
        if (cosTheta > 0.0f) {
            etai = ior;
            etat = 1.0f;
            n = -n;
            cosTheta = -cosTheta;
        }
        float eta = etai / etat;
        float k = 1.0f - eta * eta * (1.0f - cosTheta * cosTheta);
        Vector3f Nface = faceNormal(D, geomN);
        split.reflectDir = (D - 2.0f * Vector3f::dot(D, Nface) * Nface).normalized();
        if (k < 0.0f) {
            split.tir = true;
            split.refractDir = split.reflectDir;
            return split;
        }
        split.refractDir =
            (eta * D + (eta * cosTheta - sqrtf(k)) * n).normalized();
        return split;
    }

    Vector3f traceDielectricReflect(const Ray &ray, const Hit &hit, RefractMaterial *mat, int depth,
                                    const Vector3f &throughput, int dispChannel,
                                    float probWeight = 1.0f) const {
        Vector3f hitPoint = ray.pointAtParameter(hit.getT());
        Vector3f D = ray.getDirection().normalized();
        Vector3f geomN = hit.getNormal().normalized();
        RefractSplit split = computeRefractSplit(D, geomN, mat->getRefractIndex());
        Vector3f origin = offsetAlongNormal(hitPoint, faceNormal(D, geomN), ORIGIN_OFFSET);
        Vector3f newThroughput = throughput;
        if (probWeight > 1e-8f) {
            newThroughput = newThroughput / probWeight;
        }
        return clampRadiance(castRayPath(Ray(origin, split.reflectDir), depth + 1, newThroughput, true,
                                         nullptr, dispChannel, RAY_EPSILON));
    }

    Vector3f traceDielectricRefract(const Ray &ray, const Hit &hit, RefractMaterial *mat, int depth,
                                    const Vector3f &throughput, int dispChannel, float ior,
                                    float probWeight = 1.0f) const {
        Vector3f hitPoint = ray.pointAtParameter(hit.getT());
        Vector3f D = ray.getDirection().normalized();
        Vector3f geomN = hit.getNormal().normalized();
        RefractSplit split = computeRefractSplit(D, geomN, ior);
        Vector3f origin = offsetAlongRay(hitPoint, split.refractDir, REFRACT_ORIGIN_OFFSET);
        bool exiting = Vector3f::dot(D, geomN) > 0.0f;
        Vector3f newThroughput = exiting ? throughput
                                           : scaleDispAttenuation(throughput, mat->getRefractColor(), dispChannel);
        if (probWeight > 1e-8f) {
            newThroughput = newThroughput / probWeight;
        }
        return clampRadiance(castRayPath(Ray(origin, split.refractDir), depth + 1, newThroughput, true,
                                         nullptr, dispChannel, REFRACT_RAY_TMIN));
    }

    static float triangleArea(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2) {
        return 0.5f * Vector3f::cross(v1 - v0, v2 - v0).length();
    }

    Vector3f sampleTriangle(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2) const {
        float u = uniform();
        float v = uniform();
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        return v0 + (v1 - v0) * u + (v2 - v0) * v;
    }

    static void buildOrthonormalBasis(const Vector3f &n, Vector3f &tangent, Vector3f &bitangent) {
        if (fabsf(n.x()) > fabsf(n.y())) {
            tangent = Vector3f(-n.z(), 0.0f, n.x()).normalized();
        } else {
            tangent = Vector3f(0.0f, n.z(), -n.y()).normalized();
        }
        bitangent = Vector3f::cross(n, tangent);
    }

    Vector3f sampleCosineHemisphere(const Vector3f &normal, float &pdf) const {
        float u1 = uniform();
        float u2 = uniform();
        float phi = 2.0f * M_PI_F * u1;
        float cosTheta = sqrtf(u2);
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

        Vector3f tangent;
        Vector3f bitangent;
        buildOrthonormalBasis(normal, tangent, bitangent);
        pdf = cosTheta / M_PI_F;
        return tangent * (cosf(phi) * sinTheta) +
               bitangent * (sinf(phi) * sinTheta) +
               normal * cosTheta;
    }

    Vector3f sampleGGXHalfVector(const Vector3f &n, const Vector3f &wo, float roughness,
                                 float &pdf) const {
        float u1 = uniform();
        float u2 = uniform();
        float phi = 2.0f * M_PI_F * u1;
        float alpha2 = CookTorranceBRDF::alphaFromRoughness(roughness);
        alpha2 = alpha2 * alpha2;
        float cosTheta2 = (1.0f - u2) / (1.0f + (alpha2 - 1.0f) * u2);
        float cosTheta = sqrtf(fmaxf(0.0f, cosTheta2));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

        Vector3f tangent;
        Vector3f bitangent;
        buildOrthonormalBasis(n, tangent, bitangent);
        Vector3f h = tangent * (cosf(phi) * sinTheta) +
                     bitangent * (sinf(phi) * sinTheta) +
                     n * cosTheta;
        if (Vector3f::dot(h, wo) < 0.0f) {
            h = -h;
        }
        float nh = std::max(0.0f, Vector3f::dot(n, h));
        float D = CookTorranceBRDF::ggxD(n, h, roughness);
        pdf = D * nh;
        return h;
    }

    static bool rayTriangleIntersect(const Vector3f &orig, const Vector3f &dir,
                                     const Vector3f &v0, const Vector3f &v1, const Vector3f &v2,
                                     float &tOut, float &uOut, float &vOut) {
        Vector3f e1 = v1 - v0;
        Vector3f e2 = v2 - v0;
        Vector3f pvec = Vector3f::cross(dir, e2);
        float det = Vector3f::dot(e1, pvec);
        if (fabsf(det) < 1e-8f) {
            return false;
        }
        float invDet = 1.0f / det;
        Vector3f tvec = orig - v0;
        uOut = Vector3f::dot(tvec, pvec) * invDet;
        if (uOut < 0.0f || uOut > 1.0f) {
            return false;
        }
        Vector3f qvec = Vector3f::cross(tvec, e1);
        vOut = Vector3f::dot(dir, qvec) * invDet;
        if (vOut < 0.0f || uOut + vOut > 1.0f) {
            return false;
        }
        tOut = Vector3f::dot(e2, qvec) * invDet;
        return tOut > RAY_EPSILON;
    }

    float pdfDiffuseBRDF(float cosO) const {
        return cosO / M_PI_F;
    }

    float pdfGlossyBRDF(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                        const Vector3f &kd, const Vector3f &ks, float roughness) const {
        float cosO = std::max(0.0f, Vector3f::dot(n, wi));
        if (cosO <= 0.0f) {
            return 0.0f;
        }
        float kdLum = luminance(kd);
        float ksLum = luminance(ks);
        bool isMetal = kdLum < 0.01f;
        float specProb = isMetal ? 1.0f : ksLum / std::max(1e-4f, kdLum + ksLum);

        float pdfDiff = (1.0f - specProb) * cosO / M_PI_F;

        Vector3f h = (wi + wo).normalized();
        float nh = std::max(0.0f, Vector3f::dot(n, h));
        if (nh <= 0.0f) {
            return pdfDiff;
        }
        float D = CookTorranceBRDF::ggxD(n, h, roughness);
        float pdfH = D * nh;
        float vh = std::max(0.001f, Vector3f::dot(wo, h));
        float pdfSpec = specProb * pdfH / (4.0f * vh);
        float pdfTotal = pdfDiff + pdfSpec;
        return pdfTotal;
    }

    float pdfGlossyBRDF(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                        GlossyMaterial *mat) const {
        return pdfGlossyBRDF(n, wo, wi, mat->getDiffuseColor(), mat->getSpecularColor(),
                             mat->getRoughness());
    }

    float pdfAreaLightDirection(const AreaLight *area, const Vector3f &hitPoint,
                                const Vector3f &wi) const {
        const Vector3f &v0 = area->getVertex0();
        const Vector3f &v1 = area->getVertex1();
        const Vector3f &v2 = area->getVertex2();
        float t = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (!rayTriangleIntersect(hitPoint, wi, v0, v1, v2, t, u, v)) {
            return 0.0f;
        }
        Vector3f lightN = Vector3f::cross(v1 - v0, v2 - v0).normalized();
        float cosL = Vector3f::dot(lightN, -wi);
        if (cosL <= 0.0f) {
            return 0.0f;
        }
        float lightArea = triangleArea(v0, v1, v2);
        float dist2 = t * t;
        return dist2 / (lightArea * cosL);
    }

    float computeAreaLightPdf(const Vector3f &hitPoint, const Vector3f &wi) const {
        float pdf = 0.0f;
        for (int i = 0; i < scene.getNumLights(); ++i) {
            auto *area = dynamic_cast<AreaLight *>(scene.getLight(i));
            if (area != nullptr) {
                float areaPdf = pdfAreaLightDirection(area, hitPoint, wi);
                if (areaPdf > 0.0f) {
                    pdf += areaPdf;
                }
            }
        }
        return pdf;
    }

    static float misPowerDenom(float pdfLight, float pdfBrdf) {
        return pdfLight * pdfLight + pdfBrdf * pdfBrdf;
    }

    static float misWeightPower(float pdf, float pdfLight, float pdfBrdf) {
        float denom = misPowerDenom(pdfLight, pdfBrdf);
        if (denom < 1e-8f) {
            return 0.0f;
        }
        return pdf * pdf / denom;
    }

    bool isSegmentOccluded(const Vector3f &from, const Vector3f &to, const Vector3f &N) const {
        Vector3f omega_dir = to - from;
        float dist = omega_dir.length();
        if (dist < RAY_EPSILON) {
            return false;
        }
        omega_dir = omega_dir / dist;
        Vector3f shadowN = N;
        if (Vector3f::dot(shadowN, omega_dir) < 0.0f) {
            shadowN = -shadowN;
        }
        Vector3f origin = from + shadowN * SHADOW_EPSILON;
        for (int pass = 0; pass < 8; ++pass) {
            float remaining = (to - origin).length();
            if (remaining <= RAY_EPSILON) {
                return false;
            }
            Ray shadowRay(origin, omega_dir);
            Hit shadowHit;
            if (!scene.getGroup()->intersect(shadowRay, shadowHit, RAY_EPSILON)) {
                return false;
            }
            if (shadowHit.getT() >= remaining - SHADOW_EPSILON) {
                return false;
            }
            MaterialType blocker = shadowHit.getMaterial()->getType();
            if (blocker == MaterialType::EMISSIVE) {
                return false;
            }
            if (blocker == MaterialType::REFRACT) {
                Vector3f hitPoint = origin + omega_dir * shadowHit.getT();
                origin = offsetAlongRay(hitPoint, omega_dir, REFRACT_ORIGIN_OFFSET);
                continue;
            }
            return true;
        }
        return false;
    }

    // 原创性声明：参考已有代码，NEE 面积光采样公式来自课件；MIS 幂启发式按 PBRT 思路独立实现。
    // NEE direct term for one triangle area light (Lambertian).
    // L = Le * (albedo/pi) * cos(theta_o) / pdf_omega, pdf_omega = (1/A) * r^2 / cos(theta_l)
    bool sampleOneAreaLightDiffuse(const AreaLight *area, const Vector3f &surfPoint, const Vector3f &shadeN,
                                   const Vector3f &albedo, Vector3f &contrib) const {
        const Vector3f &v0 = area->getVertex0();
        const Vector3f &v1 = area->getVertex1();
        const Vector3f &v2 = area->getVertex2();
        Vector3f sampOnLight = sampleTriangle(v0, v1, v2);
        Vector3f toLight = sampOnLight - surfPoint;
        float segLenSq = toLight.squaredLength();
        float segLen = sqrtf(segLenSq);
        if (segLen < RAY_EPSILON) {
            return false;
        }
        Vector3f lightWi = toLight / segLen;

        float cosOut = Vector3f::dot(shadeN, lightWi);
        if (cosOut <= 0.0f) {
            return false;
        }

        Vector3f lightN = Vector3f::cross(v1 - v0, v2 - v0).normalized();
        float cosOnLight = Vector3f::dot(lightN, -lightWi);
        if (cosOnLight <= 0.0f) {
            return false;
        }

        if (isSegmentOccluded(surfPoint, sampOnLight, shadeN)) {
            return false;
        }

        float triArea = triangleArea(v0, v1, v2);
        Vector3f emission = area->getColor();
        if (useMIS()) {
            float pdfLight = segLenSq / (triArea * cosOnLight);
            float pdfBrdf = pdfDiffuseBRDF(cosOut);
            float misW = misWeightPower(pdfLight, pdfLight, pdfBrdf);
            if (misW < 1e-8f) {
                return false;
            }
            float brdfTerm = cosOut / M_PI_F;
            Vector3f lightTerm = albedo * emission * (brdfTerm / pdfLight);
            contrib += lightTerm * misW;
        } else {
            float geoFactor = cosOut * cosOnLight * triArea / (M_PI_F * segLenSq);
            contrib += albedo * emission * geoFactor;
        }
        return true;
    }

    bool sampleOneAreaLightGlossy(const AreaLight *area, const Vector3f &surfPoint, const Vector3f &shadeN,
                                  const Vector3f &wo, GlossyMaterial *mat, Vector3f &contrib) const {
        const Vector3f &v0 = area->getVertex0();
        const Vector3f &v1 = area->getVertex1();
        const Vector3f &v2 = area->getVertex2();
        Vector3f sampOnLight = sampleTriangle(v0, v1, v2);
        Vector3f toLight = sampOnLight - surfPoint;
        float segLenSq = toLight.squaredLength();
        float segLen = sqrtf(segLenSq);
        if (segLen < RAY_EPSILON) {
            return false;
        }
        Vector3f lightWi = toLight / segLen;

        float cosOut = Vector3f::dot(shadeN, lightWi);
        if (cosOut <= 0.0f) {
            return false;
        }

        Vector3f lightN = Vector3f::cross(v1 - v0, v2 - v0).normalized();
        float cosOnLight = Vector3f::dot(lightN, -lightWi);
        if (cosOnLight <= 0.0f) {
            return false;
        }

        if (isSegmentOccluded(surfPoint, sampOnLight, shadeN)) {
            return false;
        }

        float triArea = triangleArea(v0, v1, v2);
        Vector3f brdfVal = mat->evaluateBRDF(shadeN, wo, lightWi);
        if (useGlossyNEEMIS()) {
            float pdfLight = segLenSq / (triArea * cosOnLight);
            float pdfBrdf = pdfGlossyBRDF(shadeN, wo, lightWi, mat);
            float misW = misWeightPower(pdfLight, pdfLight, pdfBrdf);
            if (misW < 1e-8f) {
                return false;
            }
            contrib += brdfVal * area->getColor() * cosOut / pdfLight * misW;
        } else {
            contrib += brdfVal * area->getColor() * cosOut * cosOnLight * triArea / segLenSq;
        }
        return true;
    }

    Vector3f sampleDirectEmissive(const Vector3f &hitPoint, const Vector3f &N,
                                  const Vector3f &albedo) const {
        Vector3f direct = Vector3f::ZERO;
        for (int i = 0; i < scene.getNumLights(); ++i) {
            auto *area = dynamic_cast<AreaLight *>(scene.getLight(i));
            if (area != nullptr) {
                sampleOneAreaLightDiffuse(area, hitPoint, N, albedo, direct);
            }
        }
        return direct;
    }

    Vector3f sampleDirectPointLights(const Vector3f &hitPoint, const Vector3f &N,
                                     const Vector3f &albedo) const {
        Vector3f direct = Vector3f::ZERO;
        for (int i = 0; i < scene.getNumLights(); ++i) {
            auto *point = dynamic_cast<PointLight *>(scene.getLight(i));
            if (point == nullptr) {
                continue;
            }
            Vector3f toLight = point->getPosition() - hitPoint;
            float dist2 = toLight.squaredLength();
            if (dist2 < RAY_EPSILON * RAY_EPSILON) {
                continue;
            }
            Vector3f L = toLight / sqrtf(dist2);
            Vector3f lightColor;
            Vector3f dummy;
            scene.getLight(i)->getIllumination(hitPoint, dummy, lightColor);
            if (Vector3f::dot(N, L) <= 0.0f) {
                continue;
            }
            if (isInShadow(hitPoint, N, scene.getLight(i))) {
                continue;
            }
            float cosTheta = Vector3f::dot(N, L);
            direct += albedo * lightColor * cosTheta / dist2;
        }
        return direct;
    }

    Vector3f sampleDirectPointLightsBRDF(const Vector3f &hitPoint, const Vector3f &N,
                                         const Vector3f &wo, GlossyMaterial *mat) const {
        Vector3f direct = Vector3f::ZERO;
        for (int i = 0; i < scene.getNumLights(); ++i) {
            auto *point = dynamic_cast<PointLight *>(scene.getLight(i));
            if (point == nullptr) {
                continue;
            }
            Vector3f toLight = point->getPosition() - hitPoint;
            float dist2 = toLight.squaredLength();
            if (dist2 < RAY_EPSILON * RAY_EPSILON) {
                continue;
            }
            Vector3f wi = toLight / sqrtf(dist2);
            Vector3f lightColor;
            Vector3f dummy;
            scene.getLight(i)->getIllumination(hitPoint, dummy, lightColor);
            if (Vector3f::dot(N, wi) <= 0.0f) {
                continue;
            }
            if (isInShadow(hitPoint, N, scene.getLight(i))) {
                continue;
            }
            Vector3f brdf = mat->evaluateBRDF(N, wo, wi);
            direct += brdf * lightColor * Vector3f::dot(N, wi) / dist2;
        }
        return direct;
    }

    Vector3f sampleDirectEmissiveBRDF(const Vector3f &hitPoint, const Vector3f &N,
                                      const Vector3f &wo, GlossyMaterial *mat) const {
        Vector3f direct = Vector3f::ZERO;
        for (int i = 0; i < scene.getNumLights(); ++i) {
            auto *area = dynamic_cast<AreaLight *>(scene.getLight(i));
            if (area != nullptr) {
                sampleOneAreaLightGlossy(area, hitPoint, N, wo, mat, direct);
            }
        }
        return direct;
    }

    float survivalProbability(const Vector3f &throughput) const {
        return std::max(RR_MIN_SURVIVAL, luminance(throughput));
    }

    Vector3f traceReflectChild(const Ray &ray, const Hit &hit, ReflectMaterial *mat, int depth,
                               const Vector3f &throughput, int dispChannel) const {
        Vector3f hitPoint = ray.pointAtParameter(hit.getT());
        Vector3f D = ray.getDirection().normalized();
        Vector3f N = faceNormal(D, hit.getNormal());
        Vector3f reflected = D - 2.0f * Vector3f::dot(D, N) * N;
        Vector3f origin = offsetAlongNormal(hitPoint, N, ORIGIN_OFFSET);
        Vector3f newThroughput = scaleDispAttenuation(throughput, mat->getReflectColor(), dispChannel);
        return clampRadiance(castRayPath(Ray(origin, reflected), depth + 1, newThroughput, true,
                                         nullptr, dispChannel, RAY_EPSILON));
    }

    static float channelIor(float baseIor, float delta, int channel) {
        // R/G/B use base−δ/2, base, base+δ/2 — applied on exit split and along mono-channel child paths.
        if (channel == 0) {
            return baseIor - delta * 0.5f;
        }
        if (channel == 2) {
            return baseIor + delta * 0.5f;
        }
        return baseIor;
    }

    static Vector3f scaleDispAttenuation(const Vector3f &throughput, const Vector3f &attenColor,
                                         int dispChannel) {
        if (dispChannel < 0) {
            return Vector3f(
                throughput[0] * attenColor[0],
                throughput[1] * attenColor[1],
                throughput[2] * attenColor[2]);
        }
        if (dispChannel == 0) {
            return Vector3f(throughput[0] * attenColor[0], 0.0f, 0.0f);
        }
        if (dispChannel == 1) {
            return Vector3f(0.0f, throughput[1] * attenColor[1], 0.0f);
        }
        return Vector3f(0.0f, 0.0f, throughput[2] * attenColor[2]);
    }

    Vector3f traceRefractChild(const Ray &ray, const Hit &hit, RefractMaterial *mat, int depth,
                                const Vector3f &throughput, int dispChannel) const {
        Vector3f hitPoint = ray.pointAtParameter(hit.getT());
        Vector3f D = ray.getDirection().normalized();
        Vector3f geomN = hit.getNormal().normalized();
        const Vector3f &refractColor = mat->getRefractColor();
        float baseIor = mat->getRefractIndex();
        float delta = mat->getDispersionDelta();

        // Split white light into R/G/B only when exiting glass (air-boundary).
        // Entry uses a single IOR so the prism stays clear; separation projects on the screen.
        bool exiting = Vector3f::dot(D, geomN) > 0.0f;
        if (dispersionEnabled && delta > 0.0f && dispChannel < 0 && exiting) {
            Vector3f result = Vector3f::ZERO;
            for (int c = 0; c < 3; ++c) {
                float ior = channelIor(baseIor, delta, c);
                RefractSplit split = computeRefractSplit(D, geomN, ior);
                if (split.tir) {
                    Vector3f origin = offsetAlongNormal(hitPoint, faceNormal(D, geomN), ORIGIN_OFFSET);
                    Vector3f reflChild = castRayPath(Ray(origin, split.reflectDir), depth + 1, throughput, true,
                                                     nullptr, c, RAY_EPSILON);
                    result[c] = reflChild[c];
                } else {
                    Vector3f origin = offsetAlongRay(hitPoint, split.refractDir, REFRACT_ORIGIN_OFFSET);
                    Vector3f refrChild = castRayPath(Ray(origin, split.refractDir), depth + 1, throughput, true,
                                                     nullptr, c, REFRACT_RAY_TMIN);
                    result[c] = refrChild[c];
                }
            }
            return clampRadiance(result);
        }

        float ior = baseIor;
        if (dispChannel >= 0 && delta > 0.0f) {
            ior = channelIor(baseIor, delta, dispChannel);
        }
        RefractSplit split = computeRefractSplit(D, geomN, ior);
        if (split.tir) {
            return traceDielectricReflect(ray, hit, mat, depth, throughput, dispChannel);
        }
        return traceDielectricRefract(ray, hit, mat, depth, throughput, dispChannel, ior);
    }

    Vector3f castRayWhitted(const Ray &ray, int depth, float tMin) const {
        if (depth > MAX_TRACE_DEPTH) {
            return scene.getBackgroundColor();
        }

        Hit surfHit;
        if (!scene.getGroup()->intersect(ray, surfHit, tMin)) {
            return scene.getBackgroundColor();
        }

        Material *mat = surfHit.getMaterial();
        Vector3f surfPoint = ray.pointAtParameter(surfHit.getT());
        Vector3f viewDir = ray.getDirection().normalized();
        Vector3f geomN = surfHit.getNormal().normalized();

        if (mat->getType() == MaterialType::EMISSIVE) {
            return mat->getEmission();
        }

        if (isOpaqueBackFace(viewDir, geomN, mat->getType())) {
            return Vector3f::ZERO;
        }

        if (mat->getType() == MaterialType::GLOSSY) {
            return shadeGlossyWhitted(ray, surfHit, surfPoint);
        }

        if (mat->getType() == MaterialType::DIFFUSE) {
            return shadeDiffuse(ray, surfHit, surfPoint);
        }

        if (mat->getType() == MaterialType::REFLECT) {
            Vector3f shadeN = faceNormal(viewDir, geomN);
            Vector3f reflected = viewDir - 2.0f * Vector3f::dot(viewDir, shadeN) * shadeN;
            Vector3f bounceOrigin = offsetAlongNormal(surfPoint, shadeN, ORIGIN_OFFSET);
            auto *reflectMat = static_cast<ReflectMaterial *>(mat);
            Vector3f child = castRayWhitted(Ray(bounceOrigin, reflected), depth + 1, RAY_EPSILON);
            return reflectMat->getReflectColor() * child;
        }

        auto *refractMat = static_cast<RefractMaterial *>(mat);
        float baseIor = refractMat->getRefractIndex();
        float delta = refractMat->getDispersionDelta();
        const Vector3f &refractColor = refractMat->getRefractColor();

        bool exiting = Vector3f::dot(viewDir, geomN) > 0.0f;
        if (dispersionEnabled && delta > 0.0f && exiting) {
            Vector3f result = Vector3f::ZERO;
            for (int c = 0; c < 3; ++c) {
                float ior = channelIor(baseIor, delta, c);
                RefractSplit split = computeRefractSplit(viewDir, geomN, ior);
                if (split.tir) {
                    Vector3f reflOrigin = offsetAlongNormal(surfPoint, faceNormal(viewDir, geomN), ORIGIN_OFFSET);
                    Vector3f child = castRayWhitted(Ray(reflOrigin, split.reflectDir), depth + 1, RAY_EPSILON);
                    result[c] = child[c];
                } else {
                    Vector3f refrOrigin = offsetAlongRay(surfPoint, split.refractDir, REFRACT_ORIGIN_OFFSET);
                    Vector3f child = castRayWhitted(Ray(refrOrigin, split.refractDir), depth + 1, REFRACT_RAY_TMIN);
                    result[c] = child[c];
                }
            }
            return result;
        }

        RefractSplit split = computeRefractSplit(viewDir, geomN, baseIor);
        if (split.tir) {
            Vector3f reflOrigin = offsetAlongNormal(surfPoint, faceNormal(viewDir, geomN), ORIGIN_OFFSET);
            Vector3f child = castRayWhitted(Ray(reflOrigin, split.reflectDir), depth + 1, RAY_EPSILON);
            return child;
        }
        Vector3f refrOrigin = offsetAlongRay(surfPoint, split.refractDir, REFRACT_ORIGIN_OFFSET);
        Vector3f child = castRayWhitted(Ray(refrOrigin, split.refractDir), depth + 1, REFRACT_RAY_TMIN);
        return exiting ? child : refractColor * child;
    }

    // 原创性声明：参考已有代码（路径追踪通用框架），bounce 调度与 MIS 间接项独立实现。
    Vector3f castRayPath(const Ray &ray, int depth, const Vector3f &throughput,
                         bool countEmissive, const MisIndirectCtx *misCtx = nullptr,
                         int dispChannel = -1, float tMin = RAY_EPSILON) const {
        if (depth > MAX_TRACE_DEPTH) {
            return Vector3f::ZERO;
        }

        Hit surfHit;
        if (!scene.getGroup()->intersect(ray, surfHit, tMin)) {
            return Vector3f::ZERO;
        }

        Material *mat = surfHit.getMaterial();
        Vector3f surfPoint = ray.pointAtParameter(surfHit.getT());
        Vector3f viewDir = ray.getDirection().normalized();
        Vector3f geomN = surfHit.getNormal().normalized();

        if (mat->getType() == MaterialType::EMISSIVE) {
            if (countEmissive) {
                Vector3f emission = mat->getEmission();
                if (misCtx != nullptr) {
                    float pdfLight = computeAreaLightPdf(misCtx->shadingPoint, misCtx->wi);
                    float misW = misWeightPower(misCtx->pdfBrdf, pdfLight, misCtx->pdfBrdf);
                    if (misW >= 1e-8f) {
                        float scale = misW / misCtx->pdfBrdf;
                        return clampRadiance(Vector3f(
                            throughput[0] * emission[0] * scale,
                            throughput[1] * emission[1] * scale,
                            throughput[2] * emission[2] * scale));
                    }
                    return Vector3f::ZERO;
                }
                return clampRadiance(Vector3f(
                    throughput[0] * emission[0],
                    throughput[1] * emission[1],
                    throughput[2] * emission[2]));
            }
            return Vector3f::ZERO;
        }

        if (isOpaqueBackFace(viewDir, geomN, mat->getType())) {
            return Vector3f::ZERO;
        }

        if (mat->getType() == MaterialType::REFLECT) {
            return traceReflectChild(ray, surfHit, static_cast<ReflectMaterial *>(mat), depth, throughput,
                                     dispChannel);
        }

        if (mat->getType() == MaterialType::REFRACT) {
            return traceRefractChild(ray, surfHit, static_cast<RefractMaterial *>(mat), depth, throughput,
                                     dispChannel);
        }

        if (mat->getType() == MaterialType::GLOSSY) {
            return shadeGlossyPath(surfHit, surfPoint, viewDir, geomN, static_cast<GlossyMaterial *>(mat), depth,
                                   throughput, dispChannel);
        }

        return shadeDiffusePath(surfHit, surfPoint, viewDir, geomN, mat, depth, throughput, dispChannel);
    }

    Vector3f shadeDiffusePath(const Hit &surfHit, const Vector3f &surfPoint, const Vector3f &viewDir,
                              const Vector3f &geomN, Material *mat, int depth,
                              const Vector3f &throughput, int dispChannel) const {
        Vector3f shadeN = mat->getShadingNormal(surfHit, -viewDir);
        Vector3f surfaceAlbedo = mat->getShadedDiffuse(surfHit);

        Vector3f direct = Vector3f::ZERO;
        if (useNEE()) {
            direct = sampleDirectEmissive(surfPoint, shadeN, surfaceAlbedo) +
                     sampleDirectPointLights(surfPoint, shadeN, surfaceAlbedo);
        }

        float pdf = 0.0f;
        Vector3f lightWi = sampleCosineHemisphere(shadeN, pdf);
        Vector3f indirect = Vector3f::ZERO;
        if (pdf >= 1e-8f) {
            float rrProb = 1.0f;
            bool traceIndirect = true;
            if (depth >= RR_START_DEPTH) {
                rrProb = survivalProbability(throughput * surfaceAlbedo);
                traceIndirect = uniform() <= rrProb;
            }
            if (traceIndirect) {
                Vector3f bounceOrigin = offsetAlongNormal(surfPoint, shadeN, ORIGIN_OFFSET);
                MisIndirectCtx misCtx;
                const MisIndirectCtx *misPtr = nullptr;
                if (useMIS()) {
                    misCtx.pdfBrdf = pdf;
                    misCtx.wi = lightWi;
                    misCtx.shadingPoint = surfPoint;
                    misCtx.N = shadeN;
                    misCtx.wo = -viewDir;
                    misCtx.glossyMat = nullptr;
                    misPtr = &misCtx;
                }
                bool indirectEmissive = !useNEE() || useMIS();
                Vector3f Li = castRayPath(Ray(bounceOrigin, lightWi), depth + 1, throughput, indirectEmissive,
                                          misPtr, dispChannel);
                indirect = clampRadiance(Vector3f(
                    surfaceAlbedo[0] * Li[0],
                    surfaceAlbedo[1] * Li[1],
                    surfaceAlbedo[2] * Li[2]) / rrProb);
            }
        }

        return clampRadiance(Vector3f(
            throughput[0] * direct[0] + indirect[0],
            throughput[1] * direct[1] + indirect[1],
            throughput[2] * direct[2] + indirect[2]));
    }

    Vector3f shadeGlossyPath(const Hit &surfHit, const Vector3f &surfPoint, const Vector3f &viewDir,
                             const Vector3f &geomN, GlossyMaterial *mat, int depth,
                             const Vector3f &throughput, int dispChannel) const {
        Vector3f wo = -viewDir;
        Vector3f shadeN = mat->getShadingNormal(surfHit, wo);
        Vector3f kd = mat->getShadedDiffuse(surfHit);
        Vector3f ks = mat->getSpecularColor();
        float roughness = mat->getRoughness();

        Vector3f direct = Vector3f::ZERO;
        if (useNEE()) {
            direct = sampleDirectEmissiveBRDF(surfPoint, shadeN, wo, mat) +
                     sampleDirectPointLightsBRDF(surfPoint, shadeN, wo, mat);
        }

        float kdLum = luminance(kd);
        float ksLum = luminance(ks);
        bool isMetal = kdLum < 0.01f;
        float specProb = isMetal ? 1.0f : ksLum / std::max(1e-4f, kdLum + ksLum);

        Vector3f lightWi;
        float pdfDummy = 0.0f;
        bool specularLobe = isMetal || uniform() < specProb;
        Vector3f brdfVal = Vector3f::ZERO;
        if (specularLobe) {
            float pdfH = 0.0f;
            Vector3f h = sampleGGXHalfVector(shadeN, wo, roughness, pdfH);
            lightWi = (2.0f * Vector3f::dot(wo, h) * h - wo).normalized();
            if (Vector3f::dot(shadeN, lightWi) <= 0.0f) {
                return clampRadiance(throughput * direct);
            }
            brdfVal = mat->evaluateSpecular(shadeN, wo, lightWi);
        } else {
            lightWi = sampleCosineHemisphere(shadeN, pdfDummy);
            brdfVal = mat->evaluateDiffuse(shadeN, lightWi, kd);
        }
        float pdf = pdfGlossyBRDF(shadeN, wo, lightWi, mat);

        Vector3f indirect = Vector3f::ZERO;
        if (pdf >= 1e-8f) {
            float rrProb = 1.0f;
            bool traceIndirect = true;
            if (depth >= RR_START_DEPTH) {
                rrProb = survivalProbability(throughput * (kd + ks));
                traceIndirect = uniform() <= rrProb;
            }
            if (traceIndirect) {
                float cosOut = std::max(0.0f, Vector3f::dot(shadeN, lightWi));
                Vector3f bounceOrigin = offsetAlongNormal(surfPoint, shadeN, ORIGIN_OFFSET);
                MisIndirectCtx misCtx;
                const MisIndirectCtx *misPtr = nullptr;
                if (useMIS()) {
                    misCtx.pdfBrdf = pdf;
                    misCtx.wi = lightWi;
                    misCtx.shadingPoint = surfPoint;
                    misCtx.N = shadeN;
                    misCtx.wo = wo;
                    misCtx.glossyMat = mat;
                    misPtr = &misCtx;
                }
                bool indirectEmissive = !useNEE() || useMIS();
                Vector3f Li = castRayPath(Ray(bounceOrigin, lightWi), depth + 1, throughput, indirectEmissive,
                                          misPtr, dispChannel);
                indirect = clampRadiance(brdfVal * cosOut * Li / (pdf * rrProb));
            }
        }

        return clampRadiance(throughput * direct + indirect);
    }

    Vector3f shadeGlossyWhitted(const Ray &ray, const Hit &hit, const Vector3f &hitPoint) const {
        auto *glossy = static_cast<GlossyMaterial *>(hit.getMaterial());
        Vector3f radiance_val = Vector3f::ZERO;
        Vector3f V = ray.getDirection().normalized();
        Vector3f N = glossy->getShadingNormal(hit, V);
        for (int i = 0; i < scene.getNumLights(); ++i) {
            if (isDuplicateSplitAreaLight(scene, i)) {
                continue;
            }
            Light *light = scene.getLight(i);
            if (isInShadow(hitPoint, N, light)) {
                continue;
            }
            Vector3f L, lightColor;
            light->getIllumination(hitPoint, L, lightColor);
            radiance_val += glossy->Shade(ray, hit, L, lightColor);
        }
        return radiance_val;
    }

    Vector3f shadeDiffuse(const Ray &ray, const Hit &hit, const Vector3f &hitPoint) const {
        Material *mat = hit.getMaterial();
        if (mat->getType() == MaterialType::EMISSIVE) {
            return mat->getEmission();
        }

        Vector3f radiance_val = Vector3f::ZERO;
        Vector3f V = ray.getDirection().normalized();
        Vector3f N = mat->getShadingNormal(hit, V);
        for (int i = 0; i < scene.getNumLights(); ++i) {
            if (isDuplicateSplitAreaLight(scene, i)) {
                continue;
            }
            Light *light = scene.getLight(i);
            if (isInShadow(hitPoint, N, light)) {
                continue;
            }
            Vector3f L, lightColor;
            light->getIllumination(hitPoint, L, lightColor);
            radiance_val += mat->Shade(ray, hit, L, lightColor);
        }
        return radiance_val;
    }

    bool isInShadow(const Vector3f &p, const Vector3f &N, Light *light) const {
        Vector3f L, lightColor;
        light->getIllumination(p, L, lightColor);

        Vector3f shadowN = N;
        if (Vector3f::dot(shadowN, L) < 0.0f) {
            shadowN = -shadowN;
        }
        float maxT = light->getDistance(p);
        Ray shadowRay(p + shadowN * SHADOW_EPSILON, L);
        Hit shadowHit;
        if (!scene.getGroup()->intersect(shadowRay, shadowHit, RAY_EPSILON)) {
            return false;
        }
        if (shadowHit.getT() >= maxT - SHADOW_EPSILON) {
            return false;
        }

        MaterialType blocker = shadowHit.getMaterial()->getType();
        if (blocker == MaterialType::EMISSIVE) {
            return false;
        }
        if (blocker == MaterialType::REFRACT) {
            return false;
        }
        return true;
    }
};

#endif // RAYTRACER_H
