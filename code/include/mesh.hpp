#ifndef MESH_H
#define MESH_H

// PA1 已有代码
#include <vector>
#include "object3d.hpp"
#include "triangle.hpp"
#include "Vector2f.h"
#include "Vector3f.h"


class Mesh : public Object3D {

public:
    Mesh(const char *filename, Material *m);

    struct TriangleIndex {
        TriangleIndex() {
            x[0] = 0;
            x[1] = 0;
            x[2] = 0;
            vt[0] = -1;
            vt[1] = -1;
            vt[2] = -1;
            vn[0] = -1;
            vn[1] = -1;
            vn[2] = -1;
        }
        int &operator[](const int i) { return x[i]; }
        int x[3]{};
        int vt[3];
        int vn[3];
    };

    std::vector<Vector3f> v;
    std::vector<Vector2f> vt;
    std::vector<Vector3f> vn;
    std::vector<TriangleIndex> t;
    std::vector<Vector3f> n;
    bool intersect(const Ray &r, Hit &h, float tmin) override;

private:
    void computeNormal();
    static bool intersectTriangle(const Vector3f &a, const Vector3f &b, const Vector3f &c,
                                  const Ray &r, float tmin, float tmax, float &t,
                                  float &beta, float &gamma);
};

#endif
