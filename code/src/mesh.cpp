// 文件说明：OBJ 网格加载、面法线计算与 Möller–Trumbore 求交。
// 原创性声明：已有代码（PA1 Mesh 实现）基础之上，平滑法线/纹理坐标与 TBN 插值独立实现。

#include "mesh.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <cmath>
#include <cstdio>

bool Mesh::intersectTriangle(const Vector3f &a, const Vector3f &b, const Vector3f &c,
                             const Ray &r, float tmin, float tmax, float &t,
                             float &beta, float &gamma) {
    Vector3f rayOrig = r.getOrigin();
    Vector3f rayDir = r.getDirection();
    Vector3f e1 = a - b;
    Vector3f e2 = a - c;
    Vector3f s = a - rayOrig;
    Matrix3f m1(s, e1, e2);
    Matrix3f m2(rayDir, s, e2);
    Matrix3f m3(rayDir, e1, s);
    Matrix3f m4(rayDir, e1, e2);
    float detMain = m4.determinant();
    if (fabsf(detMain) < 1e-6f) {
        return false;
    }
    t = m1.determinant() / detMain;
    beta = m2.determinant() / detMain;
    gamma = m3.determinant() / detMain;
    if (beta < 0.0f || gamma < 0.0f || beta + gamma > 1.0f) {
        return false;
    }
    return t > tmin && t < tmax;
}

bool Mesh::intersect(const Ray &r, Hit &h, float tmin) {
    bool result = false;
    for (int triId = 0; triId < (int) t.size(); ++triId) {
        TriangleIndex &triIndex = t[triId];
        const Vector3f &a = v[triIndex[0]];
        const Vector3f &b = v[triIndex[1]];
        const Vector3f &c = v[triIndex[2]];

        float t = 0.0f;
        float beta = 0.0f;
        float gamma = 0.0f;
        if (!intersectTriangle(a, b, c, r, tmin, h.getT(), t, beta, gamma)) {
            continue;
        }

        h.set(t, material, n[triId]);
        float baryAlpha = 1.0f - beta - gamma;
        if (!vn.empty() && triIndex.vn[0] >= 0 && triIndex.vn[1] >= 0 && triIndex.vn[2] >= 0) {
            int nvn = (int) vn.size();
            if (triIndex.vn[0] < nvn && triIndex.vn[1] < nvn && triIndex.vn[2] < nvn) {
                Vector3f smoothN = vn[triIndex.vn[0]] * baryAlpha + vn[triIndex.vn[1]] * beta +
                                   vn[triIndex.vn[2]] * gamma;
                if (smoothN.length() > 1e-8f) {
                    h.setNormal(smoothN.normalized());
                }
            }
        }
        if (!vt.empty() && triIndex.vt[0] >= 0 && triIndex.vt[1] >= 0 && triIndex.vt[2] >= 0) {
            int nvt = (int) vt.size();
            if (triIndex.vt[0] < nvt && triIndex.vt[1] < nvt && triIndex.vt[2] < nvt) {
                const Vector2f &uv0 = vt[triIndex.vt[0]];
                const Vector2f &uv1 = vt[triIndex.vt[1]];
                const Vector2f &uv2 = vt[triIndex.vt[2]];
                Vector2f uv = uv0 * baryAlpha + uv1 * beta + uv2 * gamma;
                h.setUV(uv);

                Vector3f e1 = b - a;
                Vector3f e2 = c - a;
                Vector2f duv1 = uv1 - uv0;
                Vector2f duv2 = uv2 - uv0;
                float det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
                if (fabsf(det) > 1e-8f) {
                    float invDet = 1.0f / det;
                    Vector3f tangent = (e1 * duv2[1] - e2 * duv1[1]) * invDet;
                    Vector3f bitangent = (-e1 * duv2[0] + e2 * duv1[0]) * invDet;
                    if (tangent.length() > 1e-8f && bitangent.length() > 1e-8f) {
                        h.setTBN(tangent.normalized(), bitangent.normalized());
                    }
                }
            }
        }
        result = true;
    }
    return result;
}

Mesh::Mesh(const char *filename, Material *material) : Object3D(material) {
    std::ifstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cout << "Unable to open mesh file: " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
    std::string normTok("vn");
    std::string tok;
    while (true) {
        std::getline(f, line);
        if (f.eof()) {
            break;
        }
        if (line.size() < 3) {
            continue;
        }
        if (line.at(0) == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tok;
        if (tok == vTok) {
            Vector3f vec;
            ss >> vec[0] >> vec[1] >> vec[2];
            v.push_back(vec);
        } else if (tok == texTok) {
            Vector2f texcoord;
            ss >> texcoord[0] >> texcoord[1];
            vt.push_back(texcoord);
        } else if (tok == normTok) {
            Vector3f normal;
            ss >> normal[0] >> normal[1] >> normal[2];
            vn.push_back(normal.normalized());
        } else if (tok == fTok) {
            TriangleIndex trig;
            std::stringstream facess(line);
            facess >> tok;
            for (int ii = 0; ii < 3; ii++) {
                std::string vert;
                facess >> vert;
                int vIdx = 0;
                int vtIdx = -1;
                int vnIdx = -1;
                int matched = std::sscanf(vert.c_str(), "%d/%d/%d", &vIdx, &vtIdx, &vnIdx);
                if (matched < 1) {
                    matched = std::sscanf(vert.c_str(), "%d//%d", &vIdx, &vnIdx);
                    if (matched < 1) {
                        matched = std::sscanf(vert.c_str(), "%d/%d", &vIdx, &vtIdx);
                        if (matched < 1) {
                            std::sscanf(vert.c_str(), "%d", &vIdx);
                        }
                    }
                }
                trig[ii] = vIdx - 1;
                trig.vt[ii] = (vtIdx > 0) ? vtIdx - 1 : -1;
                trig.vn[ii] = (vnIdx > 0) ? vnIdx - 1 : -1;
            }
            t.push_back(trig);
        }
    }
    computeNormal();
    f.close();
}

void Mesh::computeNormal() {
    n.resize(t.size());
    for (int triId = 0; triId < (int) t.size(); ++triId) {
        TriangleIndex &triIndex = t[triId];
        Vector3f a = v[triIndex[1]] - v[triIndex[0]];
        Vector3f b = v[triIndex[2]] - v[triIndex[0]];
        Vector3f faceN = Vector3f::cross(a, b);
        if (faceN.length() > 1e-8f) {
            n[triId] = faceN / faceN.length();
        } else {
            n[triId] = Vector3f(0, 1, 0);
        }
    }
}
