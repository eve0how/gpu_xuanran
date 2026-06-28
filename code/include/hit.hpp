#ifndef HIT_H
#define HIT_H

// 文件说明：射线-物体交点记录 t、法线、UV 与 TBN 切线帧。
// 原创性声明：参考已有代码（PA1 Hit 类），UV/TBN 扩展字段独立实现。
#include <vecmath.h>
#include "ray.hpp"

class Material;

class Hit {
public:

    // constructors
    Hit() {
        material = nullptr;
        t = 1e38;
        hasUv = false;
        hasTbn = false;
    }

    Hit(float _t, Material *m, const Vector3f &n) {
        t = _t;
        material = m;
        normal = n;
        hasUv = false;
        hasTbn = false;
    }

    Hit(const Hit &h) {
        t = h.t;
        material = h.material;
        normal = h.normal;
        uv = h.uv;
        hasUv = h.hasUv;
        tangent = h.tangent;
        bitangent = h.bitangent;
        hasTbn = h.hasTbn;
    }

    // destructor
    ~Hit() = default;

    float getT() const {
        return t;
    }

    Material *getMaterial() const {
        return material;
    }

    const Vector3f &getNormal() const {
        return normal;
    }

    void setNormal(const Vector3f &n) {
        normal = n;
    }

    void set(float hitDist, Material *m, const Vector3f &n) {
        t = hitDist;
        material = m;
        normal = n;
        hasUv = false;
        hasTbn = false;
    }

    void setUV(const Vector2f &coord) {
        uv = coord;
        hasUv = true;
    }

    bool hasTexCoord() const {
        return hasUv;
    }

    const Vector2f &getTexCoord() const {
        return uv;
    }

    void setTBN(const Vector3f &t, const Vector3f &b) {
        tangent = t;
        bitangent = b;
        hasTbn = true;
    }

    bool hasTangentFrame() const {
        return hasTbn;
    }

    const Vector3f &getTangent() const {
        return tangent;
    }

    const Vector3f &getBitangent() const {
        return bitangent;
    }

private:
    float t;
    Material *material;
    Vector3f normal;
    Vector2f uv;
    bool hasUv = false;
    Vector3f tangent;
    Vector3f bitangent;
    bool hasTbn = false;

};

inline std::ostream &operator<<(std::ostream &os, const Hit &h) {
    os << "Hit <" << h.getT() << ", " << h.getNormal() << ">";
    return os;
}

#endif // HIT_H
