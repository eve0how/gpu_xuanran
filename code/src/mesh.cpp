#include "mesh.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <sstream>
#include <cmath>
#include <cstdio>

bool Mesh::intersectTriangle(const Vector3f &a, const Vector3f &b, const Vector3f &c,
                             const Ray &r, float tmin, float tmax, float &t,
                             float &beta, float &gamma) {
    Vector3f R_0 = r.getOrigin();
    Vector3f R_d = r.getDirection();
    Vector3f e1 = a - b;
    Vector3f e2 = a - c;
    Vector3f s = a - R_0;
    Matrix3f m1(s, e1, e2);
    Matrix3f m2(R_d, s, e2);
    Matrix3f m3(R_d, e1, s);
    Matrix3f m4(R_d, e1, e2);
    float chu = m4.determinant();
    if (fabs(chu) < 1e-6) {
        return false;
    }
    t = m1.determinant() / chu;
    beta = m2.determinant() / chu;
    gamma = m3.determinant() / chu;
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
        if (!vt.empty() && triIndex.vt[0] >= 0 && triIndex.vt[1] >= 0 && triIndex.vt[2] >= 0) {
            int nvt = (int) vt.size();
            if (triIndex.vt[0] < nvt && triIndex.vt[1] < nvt && triIndex.vt[2] < nvt) {
                float alpha = 1.0f - beta - gamma;
                const Vector2f &uv0 = vt[triIndex.vt[0]];
                const Vector2f &uv1 = vt[triIndex.vt[1]];
                const Vector2f &uv2 = vt[triIndex.vt[2]];
                Vector2f uv = uv0 * alpha + uv1 * beta + uv2 * gamma;
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
        std::cout << "Cannot open " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
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
