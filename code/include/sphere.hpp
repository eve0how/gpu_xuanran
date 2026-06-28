#ifndef SPHERE_H
#define SPHERE_H

/// PA1 已有代码
#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

class Sphere : public Object3D {
public:
    Sphere() {
        center = Vector3f(0, 0, 0);
        radius = 1.0f;
    }

    Sphere(const Vector3f &center, float radius, Material *material) : Object3D(material) {
        this->center = center;
        this->radius = radius;
    }

    ~Sphere() override = default;

    const Vector3f &getCenter() const { return center; }
    float getRadius() const { return radius; }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f oc = r.getOrigin() - center;
        Vector3f rayDir = r.getDirection();
        float a = Vector3f::dot(rayDir, rayDir);
        float b = 2.0f * Vector3f::dot(oc, rayDir);
        float c = Vector3f::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f) {
            return false;
        }

        float sqrtDisc = sqrtf(discriminant);
        float tNear = (-b - sqrtDisc) / (2.0f * a);
        float tFar = (-b + sqrtDisc) / (2.0f * a);

        float hitDist = -1.0f;
        if (tNear > tmin) {
            hitDist = tNear;
        } else if (tFar > tmin) {
            hitDist = tFar;
        }
        if (hitDist < 0.0f || hitDist >= h.getT()) {
            return false;
        }

        Vector3f normal = r.pointAtParameter(hitDist) - center;
        normal.normalize();
        h.set(hitDist, material, normal);

        Vector3f p = r.pointAtParameter(hitDist) - center;
        float u = 0.5f + atan2f(p.z(), p.x()) / (2.0f * static_cast<float>(M_PI));
        float v = 0.5f - asinf(std::max(-1.0f, std::min(1.0f, p.y() / radius))) / static_cast<float>(M_PI);
        h.setUV(Vector2f(u, v));
        float phi = atan2f(p.z(), p.x());
        Vector3f T(-sinf(phi), 0.0f, cosf(phi));
        if (T.squaredLength() < 1e-8f) {
            T = Vector3f(1.0f, 0.0f, 0.0f);
        } else {
            T.normalize();
        }
        Vector3f B = Vector3f::cross(normal, T).normalized();
        h.setTBN(T, B);
        return true;
    }

protected:
    Vector3f center;
    float radius;
};


#endif
