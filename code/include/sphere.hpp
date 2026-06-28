#ifndef SPHERE_H
#define SPHERE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

// TODO: Implement functions and add more fields as necessary
// Done
class Sphere : public Object3D {
public:
    Sphere() {
        // unit ball at the center
        this->center = Vector3f(0, 0, 0);
        this->radius = 1.0f;
    }

    Sphere(const Vector3f &center, float radius, Material *material) : Object3D(material) {
        // set the center and the radius
        this->center = center;
        this->radius = radius;
    }

    ~Sphere() override = default;

    const Vector3f &getCenter() const { return center; }
    float getRadius() const { return radius; }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f oc = r.getOrigin() - center;
        Vector3f rd = r.getDirection();
        float a = Vector3f::dot(rd, rd);
        float b = 2.0f * Vector3f::dot(oc, rd);
        float c = Vector3f::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f) {
            return false;
        }

        float sqrtDisc = sqrtf(discriminant);
        float t0 = (-b - sqrtDisc) / (2.0f * a);
        float t1 = (-b + sqrtDisc) / (2.0f * a);

        float t = -1.0f;
        if (t0 > tmin) {
            t = t0;
        } else if (t1 > tmin) {
            t = t1;
        }
        if (t < 0.0f || t >= h.getT()) {
            return false;
        }

        Vector3f normal = r.pointAtParameter(t) - center;
        normal.normalize();
        h.set(t, material, normal);

        Vector3f p = r.pointAtParameter(t) - center;
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
