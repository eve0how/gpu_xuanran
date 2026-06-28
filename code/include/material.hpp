#ifndef MATERIAL_H
#define MATERIAL_H

// 文件说明：材质类型与 Cook-Torrance BRDF、Phong 着色接口。
// 原创性声明：PA1 已有代码为基础，
// BRDF 求值与各类 Material 子类扩展为独立实现。

#include <cassert>
#include <cmath>
#include <algorithm>

#include <vecmath.h>

#include "ray.hpp"
#include "hit.hpp"
#include "texture.hpp"

constexpr float CT_PI = 3.14159265358979323846f;

enum class MaterialType {
    DIFFUSE,
    REFLECT,
    REFRACT,
    EMISSIVE,
    GLOSSY
};

// Cook-Torrance microfacet BRDF (isotropic GGX):
// f = kd * rho_d/pi + ks * D*G*F / (4*(n·wi)*(n·wo))
// D: GGX (Trowbridge-Reitz) NDF, G: Smith masking-shadowing, F: Schlick Fresnel.
struct CookTorranceBRDF {
    static float alphaFromRoughness(float roughness) {
        return roughness * roughness;
    }

    static float ggxD(const Vector3f &n, const Vector3f &h, float roughness) {
        float alpha = alphaFromRoughness(roughness);
        float alpha2 = alpha * alpha;
        float cosNH = std::max(0.0f, Vector3f::dot(n, h));
        float cosNH2 = cosNH * cosNH;
        float denom = cosNH2 * (alpha2 - 1.0f) + 1.0f;
        return alpha2 / (CT_PI * denom * denom);
    }

    static float smithGGX_G1(float cosTheta, float alpha) {
        float cosThetaClamped = std::max(0.001f, cosTheta);
        float cos2 = cosThetaClamped * cosThetaClamped;
        float tan2 = (1.0f - cos2) / cos2;
        return 2.0f / (1.0f + sqrtf(1.0f + alpha * alpha * tan2));
    }

    static float smithGGX_G(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                            float roughness) {
        float alpha = alphaFromRoughness(roughness);
        float nv = std::max(0.0f, Vector3f::dot(n, wo));
        float nl = std::max(0.0f, Vector3f::dot(n, wi));
        return smithGGX_G1(nv, alpha) * smithGGX_G1(nl, alpha);
    }

    static Vector3f schlickF(const Vector3f &f0, float cosTheta) {
        float f = powf(1.0f - cosTheta, 5.0f);
        return f0 + (Vector3f(1.0f, 1.0f, 1.0f) - f0) * f;
    }

    static Vector3f evaluateSpecular(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                                     const Vector3f &ks, float roughness, const Vector3f &f0) {
        float nl = Vector3f::dot(n, wi);
        float nv = Vector3f::dot(n, wo);
        if (nl <= 0.0f || nv <= 0.0f) {
            return Vector3f::ZERO;
        }
        Vector3f h = (wi + wo).normalized();
        float nh = Vector3f::dot(n, h);
        if (nh <= 0.0f) {
            return Vector3f::ZERO;
        }
        float D = ggxD(n, h, roughness);
        float G = smithGGX_G(n, wo, wi, roughness);
        Vector3f F = schlickF(f0, std::max(0.0f, Vector3f::dot(h, wo)));
        return ks * (D * G * F) / (4.0f * nl * nv);
    }

    static Vector3f evaluate(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                             const Vector3f &kd, const Vector3f &ks, float roughness,
                             const Vector3f &f0) {
        float nl = Vector3f::dot(n, wi);
        float nv = Vector3f::dot(n, wo);
        if (nl <= 0.0f || nv <= 0.0f) {
            return Vector3f::ZERO;
        }
        Vector3f diffuse = kd * (1.0f / CT_PI);
        Vector3f spec = evaluateSpecular(n, wo, wi, ks, roughness, f0);
        return diffuse + spec;
    }
};

// Phong diffuse material (PA1).
class Material {
public:
    explicit Material(const Vector3f &d_color, const Vector3f &s_color = Vector3f::ZERO, float s = 0) :
            diffuseColor(d_color), specularColor(s_color), shininess(s) {
    }

    virtual ~Material() {
        delete texture;
        delete normalMap;
    }

    virtual MaterialType getType() const {
        return MaterialType::DIFFUSE;
    }

    virtual Vector3f getDiffuseColor() const {
        return diffuseColor;
    }

    const Vector3f &getSpecularColor() const {
        return specularColor;
    }

    float getShininess() const {
        return shininess;
    }

    virtual Vector3f getShadedDiffuse(const Hit &hit) const {
        if (texture != nullptr && hit.hasTexCoord()) {
            return diffuseColor * texture->sample(hit.getTexCoord()[0], hit.getTexCoord()[1]);
        }
        return diffuseColor;
    }

    void setTexture(Texture *tex) {
        delete texture;
        texture = tex;
    }

    void setNormalMap(Texture *map) {
        delete normalMap;
        normalMap = map;
    }

    bool hasTexture() const {
        return texture != nullptr;
    }

    bool hasNormalMap() const {
        return normalMap != nullptr;
    }

    static Vector3f faceShadingNormal(const Vector3f &viewDir, const Vector3f &geomN) {
        Vector3f n = geomN.normalized();
        if (Vector3f::dot(viewDir, n) < 0.0f) {
            n = -n;
        }
        return n;
    }

    Vector3f getShadingNormal(const Hit &hit, const Vector3f &viewDir) const {
        Vector3f n = faceShadingNormal(viewDir, hit.getNormal());
        if (normalMap != nullptr && hit.hasTexCoord() && hit.hasTangentFrame()) {
            Vector3f mapN = normalMap->sampleNormal(hit.getTexCoord()[0], hit.getTexCoord()[1]);
            Vector3f T = hit.getTangent().normalized();
            Vector3f B = hit.getBitangent().normalized();
            Vector3f worldN = (T * mapN[0] + B * mapN[1] + n * mapN[2]).normalized();
            return faceShadingNormal(viewDir, worldN);
        }
        return n;
    }

    virtual Vector3f getEmission() const {
        return Vector3f::ZERO;
    }

    Vector3f Shade(const Ray &ray, const Hit &hit,
                   const Vector3f &dirToLight, const Vector3f &lightColor) const {
        Vector3f V = -ray.getDirection().normalized();
        Vector3f N = getShadingNormal(hit, V);
        Vector3f L = dirToLight.normalized();

        float d = Vector3f::dot(N, L);
        float diffuse = d > 0.0f ? d : 0.0f;

        Vector3f kd = getShadedDiffuse(hit);

        float s = 0.0f;
        if (d > 0.0f) {
            Vector3f R = 2.0f * d * N - L;
            float spec = Vector3f::dot(R, V);
            if (spec > 0.0f) {
                s = pow(spec, shininess);
            }
        }

        return kd * diffuse * lightColor + specularColor * s * lightColor;
    }

protected:
    Vector3f diffuseColor;
    Vector3f specularColor;
    float shininess;
    Texture *texture = nullptr;
    Texture *normalMap = nullptr;
};

class GlossyMaterial : public Material {
public:
    GlossyMaterial(const Vector3f &kd, const Vector3f &ks, float roughness,
                   const Vector3f &f0 = Vector3f(-1.0f, -1.0f, -1.0f)) :
            Material(kd, ks, 0.0f), specKs(ks), roughness(std::max(0.03f, roughness)) {
        if (f0[0] >= 0.0f) {
            F0 = f0;
        } else {
            float kdLum = 0.2126f * kd[0] + 0.7152f * kd[1] + 0.0722f * kd[2];
            F0 = (kdLum < 0.01f) ? ks : Vector3f(0.04f, 0.04f, 0.04f);
        }
    }

    MaterialType getType() const override {
        return MaterialType::GLOSSY;
    }

    Vector3f getDiffuseColor() const override {
        return diffuseColor;
    }

    const Vector3f &getSpecularColor() const {
        return specKs;
    }

    float getRoughness() const {
        return roughness;
    }

    const Vector3f &getF0() const {
        return F0;
    }

    Vector3f getShadedDiffuse(const Hit &hit) const override {
        if (texture != nullptr && hit.hasTexCoord()) {
            return diffuseColor * texture->sample(hit.getTexCoord()[0], hit.getTexCoord()[1]);
        }
        return diffuseColor;
    }

    Vector3f evaluateBRDF(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                          const Vector3f &kd) const {
        return CookTorranceBRDF::evaluate(n, wo, wi, kd, specKs, roughness, F0);
    }

    Vector3f evaluateBRDF(const Vector3f &n, const Vector3f &wo, const Vector3f &wi) const {
        return evaluateBRDF(n, wo, wi, diffuseColor);
    }

    Vector3f evaluateDiffuse(const Vector3f &n, const Vector3f &wi, const Vector3f &kd) const {
        if (Vector3f::dot(n, wi) <= 0.0f) {
            return Vector3f::ZERO;
        }
        return kd * (1.0f / CT_PI);
    }

    Vector3f evaluateDiffuse(const Vector3f &n, const Vector3f &wi) const {
        return evaluateDiffuse(n, wi, diffuseColor);
    }

    Vector3f evaluateSpecular(const Vector3f &n, const Vector3f &wo, const Vector3f &wi) const {
        return CookTorranceBRDF::evaluateSpecular(n, wo, wi, specKs, roughness, F0);
    }

    Vector3f Shade(const Ray &ray, const Hit &hit,
                   const Vector3f &dirToLight, const Vector3f &lightColor) const {
        Vector3f V = -ray.getDirection().normalized();
        Vector3f N = getShadingNormal(hit, V);
        Vector3f wi = dirToLight.normalized();
        Vector3f wo = V;
        if (Vector3f::dot(N, wi) <= 0.0f) {
            return Vector3f::ZERO;
        }
        Vector3f kd = getShadedDiffuse(hit);
        Vector3f brdf = evaluateBRDF(N, wo, wi, kd);
        return brdf * lightColor * Vector3f::dot(N, wi);
    }

private:
    Vector3f specKs;
    float roughness;
    Vector3f F0;
};

class ReflectMaterial : public Material {
public:
    explicit ReflectMaterial(const Vector3f &color) :
            Material(Vector3f::ZERO), reflectColor(color) {
    }

    MaterialType getType() const override {
        return MaterialType::REFLECT;
    }

    const Vector3f &getReflectColor() const {
        return reflectColor;
    }

private:
    Vector3f reflectColor;
};

class RefractMaterial : public Material {
public:
    RefractMaterial(const Vector3f &color, float index, float dispersion = 0.0f) :
            Material(Vector3f::ZERO), refractColor(color), refractIndex(index),
            dispersionDelta(dispersion) {
    }

    MaterialType getType() const override {
        return MaterialType::REFRACT;
    }

    const Vector3f &getRefractColor() const {
        return refractColor;
    }

    float getRefractIndex() const {
        return refractIndex;
    }

    float getDispersionDelta() const {
        return dispersionDelta;
    }

private:
    Vector3f refractColor;
    float refractIndex;
    float dispersionDelta;
};

class EmissiveMaterial : public Material {
public:
    explicit EmissiveMaterial(const Vector3f &e) :
            Material(Vector3f::ZERO), emission(e) {
    }

    MaterialType getType() const override {
        return MaterialType::EMISSIVE;
    }

    Vector3f getEmission() const override {
        return emission;
    }

private:
    Vector3f emission;
};

#endif // MATERIAL_H
