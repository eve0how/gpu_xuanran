#ifndef OBJECT3D_H
#define OBJECT3D_H

// PA1 已有代码
#include "ray.hpp"
#include "hit.hpp"
#include "material.hpp"

class Object3D {
public:
    Object3D() : material(nullptr) {}

    virtual ~Object3D() = default;

    explicit Object3D(Material *material) : material(material) {}

    virtual bool intersect(const Ray &r, Hit &h, float tmin) = 0;

    Material *getMaterial() const { return material; }

protected:

    Material *material;
};

#endif

