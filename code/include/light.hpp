#ifndef LIGHT_H
#define LIGHT_H

#include <cfloat>
#include <Vector3f.h>
#include "object3d.hpp"

class Light {
public:
    Light() = default;

    virtual ~Light() = default;

    virtual void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const = 0;

    // Upper bound for shadow-ray intersection (point lights only).
    virtual float getDistance(const Vector3f &p) const {
        (void) p;
        return FLT_MAX;
    }
};


class DirectionalLight : public Light {
public:
    DirectionalLight() = delete;

    DirectionalLight(const Vector3f &d, const Vector3f &c) {
        direction = d.normalized();
        color = c;
    }

    ~DirectionalLight() override = default;

    ///@param p unsed in this function
    ///@param distanceToLight not well defined because it's not a point light
    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = -direction;
        col = color;
    }

private:

    Vector3f direction;
    Vector3f color;

};

class PointLight : public Light {
public:
    PointLight() = delete;

    PointLight(const Vector3f &p, const Vector3f &c) {
        position = p;
        color = c;
    }

    ~PointLight() override = default;

    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        dir = (position - p);
        dir = dir / dir.length();
        col = color;
    }

    const Vector3f &getPosition() const { return position; }

    float getDistance(const Vector3f &p) const override {
        return (position - p).length();
    }

private:

    Vector3f position;
    Vector3f color;

};

// Triangle area light (path tracing; geometry should use EmissiveMaterial).
class AreaLight : public Light {
public:
    AreaLight() = delete;

    AreaLight(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2, const Vector3f &c) :
            vertex0(v0), vertex1(v1), vertex2(v2), color(c) {
    }

    ~AreaLight() override = default;

    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        Vector3f center = (vertex0 + vertex1 + vertex2) / 3.0f;
        dir = center - p;
        dir = dir / dir.length();
        col = color;
    }

    const Vector3f &getVertex0() const { return vertex0; }
    const Vector3f &getVertex1() const { return vertex1; }
    const Vector3f &getVertex2() const { return vertex2; }
    const Vector3f &getColor() const { return color; }

private:
    Vector3f vertex0;
    Vector3f vertex1;
    Vector3f vertex2;
    Vector3f color;
};

#endif // LIGHT_H
