#ifndef TRANSFORM_H
#define TRANSFORM_H
// PA1 已有代码
#include <vecmath.h>
#include "object3d.hpp"

// transforms a 3D point using a matrix, returning a 3D point
static Vector3f transformPoint(const Matrix4f &mat, const Vector3f &point) {
    return (mat * Vector4f(point, 1)).xyz();
}

// transform a 3D direction using a matrix, returning a direction
static Vector3f transformDirection(const Matrix4f &mat, const Vector3f &dir) {
    return (mat * Vector4f(dir, 0)).xyz();
}

class Transform : public Object3D {
public:
    Transform() {}

    Transform(const Matrix4f &m, Object3D *obj) : o(obj) {
        transform = m.inverse();
    }

    ~Transform() override = default;


    Object3D *getChild() const { return o; }

    Matrix4f getForwardMatrix() const { return transform.inverse(); }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f localOrig = transformPoint(transform, r.getOrigin());
        Vector3f localDir = transformDirection(transform, r.getDirection());
        Ray localRay(localOrig, localDir);
        bool inter = o->intersect(localRay, h, tmin);
        if (inter) {
            // Only update normal/TBN; h.set() would clear interpolated UV from mesh hits.
            h.setNormal(transformDirection(transform.transposed(), h.getNormal()).normalized());
            if (h.hasTangentFrame()) {
                h.setTBN(transformDirection(transform, h.getTangent()).normalized(),
                         transformDirection(transform, h.getBitangent()).normalized());
            }
        }
        return inter;
    }

protected:
    Object3D *o; //un-transformed object
    Matrix4f transform;
};

#endif //TRANSFORM_H
