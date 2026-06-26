#ifndef PLANE_H
#define PLANE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

// TODO: Implement Plane representing an infinite plane
// function: ax+by+cz=d
// choose your representation , add more fields and fill in the functions
// Done
class Plane : public Object3D {
public:
    Plane() {
        n = Vector3f(0, 1, 0);
        D = 0.0f;
    }

    Plane(const Vector3f &normal, float d, Material *m) : Object3D(m) {
        n = normal;;
        n.normalize();
        D = d;
    }

    ~Plane() override = default;

    const Vector3f &getNormal() const { return n; }
    float getOffset() const { return D; }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f R_0 = r.getOrigin();
        Vector3f R_d = r.getDirection();
        float fenmu = Vector3f::dot(n, R_d);
        if (fabs(fenmu) < 1e-6) return false;
        float t = (D - Vector3f::dot(n, R_0)) / fenmu;
        if (t > tmin && t < h.getT()){
            h.set(t, material, n);
            Vector3f p = r.pointAtParameter(t);
            Vector3f tangent;
            if (fabsf(n.y()) < 0.9f) {
                tangent = Vector3f::cross(Vector3f(0, 1, 0), n).normalized();
            } else {
                tangent = Vector3f::cross(Vector3f(0, 0, 1), n).normalized();
            }
            Vector3f bitangent = Vector3f::cross(n, tangent).normalized();
            float u = Vector3f::dot(p, tangent) * 2.0f;
            float v = Vector3f::dot(p, bitangent) * 2.0f;
            u -= floorf(u);
            v -= floorf(v);
            h.setUV(Vector2f(u, v));
            h.setTBN(tangent, bitangent);
            return true;
        }
        return false;
    }

protected:
    Vector3f n;
    float D;

};

#endif //PLANE_H
		

