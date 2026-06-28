#ifndef MATERIAL_H
#define MATERIAL_H

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
    GLOSSY,
    WARD
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
        float nh = std::max(0.0f, Vector3f::dot(n, h));
        float cos2 = nh * nh;
        float denom = cos2 * (alpha2 - 1.0f) + 1.0f;
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

// Ward anisotropic microfacet BRDF (Gregory J. Ward, 1992):
// f = rho_d/pi + rho_s/(4*pi*alpha_x*alpha_y*cos(theta_i)*cos(theta_r))
//     * exp[-tan^2(delta) / (cos^2(phi_h)/alpha_x^2 + sin^2(phi_h)/alpha_y^2)]
// alpha_x = alpha_y gives isotropic Ward (exponent depends only on tan^2 delta, i.e. 1D in delta).
struct WardBRDF {
    static constexpr float kCosThreshold = 1e-4f;

    static float specularWeightMax(const Vector3f &ks) {
        return std::max(ks[0], std::max(ks[1], ks[2]));
    }

    static Vector3f energyConservingDiffuse(const Vector3f &kd, const Vector3f &ks) {
        return kd * std::max(0.0f, 1.0f - specularWeightMax(ks));
    }

    // Lobe selection uses raw rho_d (not energy-conserved kd') so satin/brushed metals
    // with visible diffuse base get balanced diffuse vs specular path samples.
    static void lobeSamplingWeights(const Vector3f &kd, const Vector3f &ks,
                                    bool &isMetal, float &specProb) {
        float kdLum = 0.2126f * kd[0] + 0.7152f * kd[1] + 0.0722f * kd[2];
        float ksLum = 0.2126f * ks[0] + 0.7152f * ks[1] + 0.0722f * ks[2];
        isMetal = kdLum < 0.01f;
        if (isMetal) {
            specProb = 1.0f;
        } else {
            specProb = ksLum / std::max(1e-4f, kdLum + ksLum);
            specProb = std::min(specProb, 0.55f);
        }
    }
    // Deterministic tangent field on spheres: X = normalize(cross(up, N)), Y = N x X.
    static void buildConsistentBasis(const Vector3f &n, Vector3f &X, Vector3f &Y) {
        Vector3f up(0.0f, 1.0f, 0.0f);
        if (fabsf(Vector3f::dot(n, up)) > 0.99f) {
            up = Vector3f(1.0f, 0.0f, 0.0f);
        }
        X = Vector3f::cross(up, n).normalized();
        Y = Vector3f::cross(n, X).normalized();
    }

    // Scene tangent rotates the consistent basis in the tangent plane (never random buildBasis).
    static void buildFrame(const Vector3f &n, const Vector3f &tangentHint,
                           Vector3f &T, Vector3f &B) {
        Vector3f hint = tangentHint - n * Vector3f::dot(n, tangentHint);
        if (hint.squaredLength() < 1e-10f) {
            buildConsistentBasis(n, T, B);
            return;
        }
        hint.normalize();
        Vector3f X0, Y0;
        buildConsistentBasis(n, X0, Y0);
        float cosA = Vector3f::dot(X0, hint);
        float sinA = Vector3f::dot(Y0, hint);
        T = X0 * cosA + Y0 * sinA;
        B = Vector3f::cross(n, T).normalized();
    }

    static float wardExponent(const Vector3f &h, const Vector3f &n,
                              const Vector3f &T, const Vector3f &B,
                              float alphaX, float alphaY) {
        float nh = std::max(0.0f, Vector3f::dot(n, h));
        if (nh <= 1e-6f) {
            return 0.0f;
        }
        float hx = Vector3f::dot(T, h);
        float hy = Vector3f::dot(B, h);
        float tan2 = (hx * hx + hy * hy) / (nh * nh);
        float len2 = hx * hx + hy * hy;
        float cosPhi = (len2 > 1e-10f) ? hx / sqrtf(len2) : 1.0f;
        float sinPhi = (len2 > 1e-10f) ? hy / sqrtf(len2) : 0.0f;
        float denom = cosPhi * cosPhi / (alphaX * alphaX) +
                      sinPhi * sinPhi / (alphaY * alphaY);
        return expf(-tan2 / std::max(1e-10f, denom));
    }

    static float wardD(const Vector3f &h, const Vector3f &n, const Vector3f &T, const Vector3f &B,
                       float alphaX, float alphaY) {
        float nh = std::max(0.0f, Vector3f::dot(n, h));
        if (nh <= 1e-6f) {
            return 0.0f;
        }
        float expTerm = wardExponent(h, n, T, B, alphaX, alphaY);
        return expTerm / (4.0f * CT_PI * alphaX * alphaY * nh * nh * nh);
    }

    static Vector3f evaluateSpecular(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                                     const Vector3f &ks, float alphaX, float alphaY,
                                     const Vector3f &T, const Vector3f &B) {
        float nl = Vector3f::dot(n, wi);
        float nv = Vector3f::dot(n, wo);
        if (nl <= kCosThreshold || nv <= kCosThreshold) {
            return Vector3f::ZERO;
        }
        Vector3f h = (wi + wo).normalized();
        float nh = Vector3f::dot(n, h);
        if (nh <= kCosThreshold) {
            return Vector3f::ZERO;
        }
        float expTerm = wardExponent(h, n, T, B, alphaX, alphaY);
        float denom = 4.0f * CT_PI * alphaX * alphaY * nl * nv;
        return ks * (expTerm / denom);
    }

    static Vector3f evaluate(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                             const Vector3f &kd, const Vector3f &ks, float alphaX, float alphaY,
                             const Vector3f &T, const Vector3f &B) {
        float nl = Vector3f::dot(n, wi);
        float nv = Vector3f::dot(n, wo);
        if (nl <= kCosThreshold || nv <= kCosThreshold) {
            return Vector3f::ZERO;
        }
        Vector3f kdEff = energyConservingDiffuse(kd, ks);
        Vector3f diffuse = kdEff * (1.0f / CT_PI);
        Vector3f spec = evaluateSpecular(n, wo, wi, ks, alphaX, alphaY, T, B);
        return diffuse + spec;
    }

    // Ward importance sampling: phi_h = atan2(ay*sin(2pi*u1), ax*cos(2pi*u1)); pdf_h = D*cos(theta_h).
    static Vector3f sampleHalfVector(const Vector3f &n, const Vector3f &wo,
                                     float alphaX, float alphaY,
                                     const Vector3f &T, const Vector3f &B,
                                     float u1, float u2, float &pdfH) {
        float phiH = atan2f(alphaY * sinf(2.0f * CT_PI * u1),
                            alphaX * cosf(2.0f * CT_PI * u1));
        float cosPhi = cosf(phiH);
        float sinPhi = sinf(phiH);
        float logTerm = -logf(std::max(1e-6f, 1.0f - u2));
        float tanTheta2 = logTerm / (cosPhi * cosPhi / (alphaX * alphaX) +
                                     sinPhi * sinPhi / (alphaY * alphaY));
        float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
        float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));

        Vector3f h = (T * (sinTheta * cosPhi) + B * (sinTheta * sinPhi) + n * cosTheta).normalized();
        if (Vector3f::dot(h, wo) < 0.0f) {
            h = -h;
        }
        float nh = std::max(0.0f, Vector3f::dot(n, h));
        pdfH = wardD(h, n, T, B, alphaX, alphaY) * nh;
        return h;
    }
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

class WardMaterial : public Material {
public:
    WardMaterial(const Vector3f &kd, const Vector3f &ks, float alphaX, float alphaY,
                 const Vector3f &tangentDir = Vector3f(1.0f, 0.0f, 0.0f)) :
            Material(kd, ks, 0.0f), specKs(ks),
            alphaX(std::max(0.04f, alphaX)), alphaY(std::max(0.04f, alphaY)),
            tangentDirection(tangentDir) {
    }

    MaterialType getType() const override {
        return MaterialType::WARD;
    }

    Vector3f getDiffuseColor() const override {
        return diffuseColor;
    }

    const Vector3f &getSpecularColor() const {
        return specKs;
    }

    float getAlphaX() const {
        return alphaX;
    }

    float getAlphaY() const {
        return alphaY;
    }

    const Vector3f &getTangentDirection() const {
        return tangentDirection;
    }

    void buildShadingFrame(const Vector3f &n, Vector3f &T, Vector3f &B) const {
        WardBRDF::buildFrame(n, tangentDirection, T, B);
    }

    Vector3f getShadedDiffuse(const Hit &hit) const override {
        if (texture != nullptr && hit.hasTexCoord()) {
            return diffuseColor * texture->sample(hit.getTexCoord()[0], hit.getTexCoord()[1]);
        }
        return diffuseColor;
    }

    Vector3f evaluateBRDF(const Vector3f &n, const Vector3f &wo, const Vector3f &wi,
                          const Vector3f &kd) const {
        Vector3f T, B;
        buildShadingFrame(n, T, B);
        return WardBRDF::evaluate(n, wo, wi, kd, specKs, alphaX, alphaY, T, B);
    }

    Vector3f evaluateBRDF(const Vector3f &n, const Vector3f &wo, const Vector3f &wi) const {
        return evaluateBRDF(n, wo, wi, diffuseColor);
    }

    Vector3f getEnergyConservingDiffuse(const Vector3f &kd) const {
        return WardBRDF::energyConservingDiffuse(kd, specKs);
    }

    Vector3f evaluateDiffuse(const Vector3f &n, const Vector3f &wi, const Vector3f &kd) const {
        if (Vector3f::dot(n, wi) <= WardBRDF::kCosThreshold) {
            return Vector3f::ZERO;
        }
        return getEnergyConservingDiffuse(kd) * (1.0f / CT_PI);
    }

    Vector3f evaluateDiffuse(const Vector3f &n, const Vector3f &wi) const {
        return evaluateDiffuse(n, wi, diffuseColor);
    }

    Vector3f evaluateSpecular(const Vector3f &n, const Vector3f &wo, const Vector3f &wi) const {
        Vector3f T, B;
        buildShadingFrame(n, T, B);
        return WardBRDF::evaluateSpecular(n, wo, wi, specKs, alphaX, alphaY, T, B);
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
    float alphaX;
    float alphaY;
    Vector3f tangentDirection;
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
    RefractMaterial(const Vector3f &color, float index, float dispersion = 0.0f,
                    bool fresnel = true) :
            Material(Vector3f::ZERO), refractColor(color), refractIndex(index),
            dispersionDelta(dispersion), fresnelEnabled(fresnel) {
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

    bool isFresnelEnabled() const {
        return fresnelEnabled;
    }

private:
    Vector3f refractColor;
    float refractIndex;
    float dispersionDelta;
    bool fresnelEnabled;
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
