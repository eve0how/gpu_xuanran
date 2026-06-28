// 文件说明：将 SceneParser 场景树展平为 GPU 可用的 SOA 数组，并可选构建 BVH。
// 原创性声明：参考已有代码（PA1 场景对象层次与 CUDA 上传模式），
// 各类几何体变换与材质映射逻辑为按作业接口独立实现。

#include "cuda_types.h"
#include "bvh_builder.hpp"
#include "scene_parser.hpp"

namespace {
bool g_gpuSceneBuildUseBVH = true;
}

void setGpuSceneBuildUseBVH(bool use) { g_gpuSceneBuildUseBVH = use; }

#include "group.hpp"
#include "sphere.hpp"
#include "plane.hpp"
#include "triangle.hpp"
#include "mesh.hpp"
#include "transform.hpp"
#include "light.hpp"
#include "material.hpp"
#include "camera.hpp"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>

static void copyVec3ToGpu(const Vector3f &v, float out[3]) {
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
}

static Vector3f transformPointMat(const Matrix4f &m, const Vector3f &p) {
    return (m * Vector4f(p, 1.0f)).xyz();
}

static float extractUniformScale(const Matrix4f &m) {
    Vector3f col0(m(0, 0), m(1, 0), m(2, 0));
    return col0.length();
}

class SceneFlattener {
public:
    // Walk the CPU scene tree once and bake world-space SOA arrays — GPU kernels can't virtual-dispatch.
    explicit SceneFlattener(const SceneParser &scene) : scene(scene) {}

    void flattenInto(std::vector<GpuMaterial> &outMaterials,
                     std::vector<GpuSphere> &outSpheres,
                     std::vector<GpuPlane> &outPlanes,
                     std::vector<GpuTriangle> &outTriangles,
                     std::vector<GpuAreaLight> &outAreaLights,
                     std::vector<GpuPointLight> &outPointLights,
                     std::vector<GpuDirectionalLight> &outDirectionalLights,
                     GpuCamera &outCamera,
                     float outBboxMin[3],
                     float outBboxMax[3],
                     bool &outHasBbox) {
        materials = &outMaterials;
        spheres = &outSpheres;
        planes = &outPlanes;
        triangles = &outTriangles;
        areaLights = &outAreaLights;
        pointLights = &outPointLights;
        directionalLights = &outDirectionalLights;
        camera = &outCamera;

        outMaterials.clear();
        outSpheres.clear();
        outPlanes.clear();
        outTriangles.clear();
        outAreaLights.clear();
        outPointLights.clear();
        outDirectionalLights.clear();

        buildMaterialMap();
        buildMaterials();
        buildLights();
        buildCamera();
        flattenObject(scene.getGroup(), Matrix4f::identity());

        if (hasBbox) {
            Vector3f extent = bboxMax - bboxMin;
            float pad = std::max(0.05f, 0.05f * extent.length());
            bboxMin -= Vector3f(pad, pad, pad);
            bboxMax += Vector3f(pad, pad, pad);
            outBboxMin[0] = bboxMin.x();
            outBboxMin[1] = bboxMin.y();
            outBboxMin[2] = bboxMin.z();
            outBboxMax[0] = bboxMax.x();
            outBboxMax[1] = bboxMax.y();
            outBboxMax[2] = bboxMax.z();
            outHasBbox = true;
        } else {
            outHasBbox = false;
        }
    }

private:
    const SceneParser &scene;
    std::unordered_map<Material *, int> matMap;
    std::vector<GpuMaterial> *materials = nullptr;
    std::vector<GpuSphere> *spheres = nullptr;
    std::vector<GpuPlane> *planes = nullptr;
    std::vector<GpuTriangle> *triangles = nullptr;
    std::vector<GpuAreaLight> *areaLights = nullptr;
    std::vector<GpuPointLight> *pointLights = nullptr;
    std::vector<GpuDirectionalLight> *directionalLights = nullptr;
    GpuCamera *camera = nullptr;
    Vector3f bboxMin{1e30f, 1e30f, 1e30f};
    Vector3f bboxMax{-1e30f, -1e30f, -1e30f};
    bool hasBbox = false;

    void growBbox(const Vector3f &p) {
        hasBbox = true;
        bboxMin.x() = std::min(bboxMin.x(), p.x());
        bboxMin.y() = std::min(bboxMin.y(), p.y());
        bboxMin.z() = std::min(bboxMin.z(), p.z());
        bboxMax.x() = std::max(bboxMax.x(), p.x());
        bboxMax.y() = std::max(bboxMax.y(), p.y());
        bboxMax.z() = std::max(bboxMax.z(), p.z());
    }

    void growBboxTriangle(const Vector3f &a, const Vector3f &b, const Vector3f &c) {
        growBbox(a);
        growBbox(b);
        growBbox(c);
    }

    void buildMaterialMap() {
        for (int i = 0; i < scene.getNumMaterials(); ++i) {
            matMap[scene.getMaterial(i)] = i;
        }
    }

    int lookupMaterialId(Material *mat) const {
        if (mat == nullptr) {
            return 0;
        }
        auto it = matMap.find(mat);
        return it != matMap.end() ? it->second : 0;
    }

    void buildMaterials() {
        materials->resize(scene.getNumMaterials());
        for (int i = 0; i < scene.getNumMaterials(); ++i) {
            Material *mat = scene.getMaterial(i);
            GpuMaterial g{};
            g.type = static_cast<int>(mat->getType());
            copyVec3ToGpu(mat->getDiffuseColor(), g.diffuse);
            g.emission[0] = g.emission[1] = g.emission[2] = 0.0f;
            g.specular[0] = g.specular[1] = g.specular[2] = 0.0f;
            g.ior = 1.0f;
            g.roughness = 0.2f;
            g.f0[0] = g.f0[1] = g.f0[2] = 0.04f;
            g.dispersionDelta = 0.0f;
            g.shininess = 0.0f;
            copyVec3ToGpu(mat->getSpecularColor(), g.specular);
            g.shininess = mat->getShininess();

            switch (mat->getType()) {
                case MaterialType::REFLECT: {
                    auto *rm = static_cast<ReflectMaterial *>(mat);
                    copyVec3ToGpu(rm->getReflectColor(), g.specular);
                    break;
                }
                case MaterialType::REFRACT: {
                    auto *rm = static_cast<RefractMaterial *>(mat);
                    copyVec3ToGpu(rm->getRefractColor(), g.specular);
                    g.ior = rm->getRefractIndex();
                    g.dispersionDelta = rm->getDispersionDelta();
                    break;
                }
                case MaterialType::EMISSIVE:
                    copyVec3ToGpu(mat->getEmission(), g.emission);
                    break;
                case MaterialType::GLOSSY: {
                    auto *gm = static_cast<GlossyMaterial *>(mat);
                    copyVec3ToGpu(gm->getSpecularColor(), g.specular);
                    g.roughness = gm->getRoughness();
                    copyVec3ToGpu(gm->getF0(), g.f0);
                    break;
                }
                default:
                    break;
            }
            (*materials)[i] = g;
        }
    }

    void buildLights() {
        for (int i = 0; i < scene.getNumLights(); ++i) {
            if (auto *area = dynamic_cast<AreaLight *>(scene.getLight(i))) {
                GpuAreaLight g{};
                copyVec3ToGpu(area->getVertex0(), g.v0);
                copyVec3ToGpu(area->getVertex1(), g.v1);
                copyVec3ToGpu(area->getVertex2(), g.v2);
                copyVec3ToGpu(area->getColor(), g.color);
                areaLights->push_back(g);
                growBboxTriangle(area->getVertex0(), area->getVertex1(), area->getVertex2());
            } else if (auto *point = dynamic_cast<PointLight *>(scene.getLight(i))) {
                GpuPointLight g{};
                copyVec3ToGpu(point->getPosition(), g.pos);
                Vector3f dir, col;
                point->getIllumination(Vector3f::ZERO, dir, col);
                copyVec3ToGpu(col, g.color);
                pointLights->push_back(g);
            } else if (auto *dirLight = dynamic_cast<DirectionalLight *>(scene.getLight(i))) {
                GpuDirectionalLight g{};
                Vector3f L, col;
                dirLight->getIllumination(Vector3f::ZERO, L, col);
                copyVec3ToGpu(L, g.direction);
                copyVec3ToGpu(col, g.color);
                directionalLights->push_back(g);
            }
        }
    }

    void buildCamera() {
        Camera *cam = scene.getCamera();
        copyVec3ToGpu(cam->getCenter(), camera->center);
        copyVec3ToGpu(cam->getDirection(), camera->direction);
        copyVec3ToGpu(cam->getHorizontal(), camera->horizontal);
        copyVec3ToGpu(cam->getUp(), camera->up);
        copyVec3ToGpu(scene.getBackgroundColor(), camera->bg);
        auto *persp = dynamic_cast<PerspectiveCamera *>(cam);
        if (persp != nullptr) {
            persp->getProjectionParams(camera->cx, camera->cy, camera->fx, camera->fy);
        } else {
            camera->cx = cam->getWidth() * 0.5f;
            camera->cy = cam->getHeight() * 0.5f;
            camera->fx = camera->fy = cam->getHeight() * 0.5f;
        }
    }

    void addTriangle(const Vector3f &a, const Vector3f &b, const Vector3f &c, Material *mat) {
        GpuTriangle tri{};
        copyVec3ToGpu(a, tri.v0);
        copyVec3ToGpu(b, tri.v1);
        copyVec3ToGpu(c, tri.v2);
        tri.matId = lookupMaterialId(mat);
        triangles->push_back(tri);
    }

    void flattenObject(Object3D *obj, const Matrix4f &world) {
        if (obj == nullptr) {
            return;
        }
        if (auto *group = dynamic_cast<Group *>(obj)) {
            for (int childIdx = 0; childIdx < group->getGroupSize(); ++childIdx) {
                flattenObject(group->getObject(childIdx), world);
            }
            return;
        }
        if (auto *xf = dynamic_cast<Transform *>(obj)) {
            flattenObject(xf->getChild(), world * xf->getForwardMatrix());
            return;
        }
        if (auto *sphere = dynamic_cast<Sphere *>(obj)) {
            GpuSphere g{};
            Vector3f c = transformPointMat(world, sphere->getCenter());
            copyVec3ToGpu(c, g.center);
            g.radius = sphere->getRadius() * extractUniformScale(world);
            g.matId = lookupMaterialId(sphere->getMaterial());
            spheres->push_back(g);
            float r = g.radius;
            growBbox(c + Vector3f(r, r, r));
            growBbox(c - Vector3f(r, r, r));
            return;
        }
        if (auto *plane = dynamic_cast<Plane *>(obj)) {
            GpuPlane g{};
            Vector3f n = (world * Vector4f(plane->getNormal(), 0.0f)).xyz().normalized();
            Vector3f p = transformPointMat(world, plane->getNormal() * plane->getOffset());
            copyVec3ToGpu(n, g.normal);
            g.offset = Vector3f::dot(n, p);
            g.matId = lookupMaterialId(plane->getMaterial());
            planes->push_back(g);
            // Large finite slab for plane bbox (Cornell walls).
            Vector3f absN(fabsf(n.x()), fabsf(n.y()), fabsf(n.z()));
            if (absN.y() > 0.9f) {
                growBbox(Vector3f(-2.0f, p.y(), -2.0f));
                growBbox(Vector3f(2.0f, p.y(), 2.0f));
            } else if (absN.x() > 0.9f) {
                growBbox(Vector3f(p.x(), -2.0f, -2.0f));
                growBbox(Vector3f(p.x(), 2.0f, 2.0f));
            } else {
                growBbox(Vector3f(-2.0f, -2.0f, p.z()));
                growBbox(Vector3f(2.0f, 2.0f, p.z()));
            }
            return;
        }
        if (auto *tri = dynamic_cast<Triangle *>(obj)) {
            Vector3f a = transformPointMat(world, tri->vertices[0]);
            Vector3f b = transformPointMat(world, tri->vertices[1]);
            Vector3f c = transformPointMat(world, tri->vertices[2]);
            addTriangle(a, b, c, tri->getMaterial());
            growBboxTriangle(a, b, c);
            return;
        }
        if (auto *mesh = dynamic_cast<Mesh *>(obj)) {
            for (const auto &idx : mesh->t) {
                Vector3f a = transformPointMat(world, mesh->v[idx.x[0]]);
                Vector3f b = transformPointMat(world, mesh->v[idx.x[1]]);
                Vector3f c = transformPointMat(world, mesh->v[idx.x[2]]);
                addTriangle(a, b, c, mesh->getMaterial());
                growBboxTriangle(a, b, c);
            }
        }
    }
};

GpuSceneHost buildGpuSceneHost(const SceneParser &scene) {
    static std::vector<GpuMaterial> materials;
    static std::vector<GpuSphere> spheres;
    static std::vector<GpuPlane> planes;
    static std::vector<GpuTriangle> triangles;
    static std::vector<GpuBVHNode> bvhNodes;
    static std::vector<GpuTriangle> bvhTriangles;
    static std::vector<GpuAreaLight> areaLights;
    static std::vector<GpuPointLight> pointLights;
    static std::vector<GpuDirectionalLight> directionalLights;
    static GpuCamera camera{};

    GpuSceneHost host{};
    SceneFlattener flattener(scene);
    flattener.flattenInto(materials, spheres, planes, triangles, areaLights, pointLights,
                          directionalLights, camera, host.bboxMin, host.bboxMax, host.hasBbox);

    if (!triangles.empty() && g_gpuSceneBuildUseBVH) {
        buildBVH(triangles, bvhNodes, bvhTriangles);
    } else {
        bvhNodes.clear();
        bvhTriangles.clear();
    }

    host.numMaterials = static_cast<int>(materials.size());
    host.materials = materials.empty() ? nullptr : materials.data();
    host.numSpheres = static_cast<int>(spheres.size());
    host.spheres = spheres.empty() ? nullptr : spheres.data();
    host.numPlanes = static_cast<int>(planes.size());
    host.planes = planes.empty() ? nullptr : planes.data();
    host.numTriangles = static_cast<int>(triangles.size());
    host.triangles = triangles.empty() ? nullptr : triangles.data();
    host.numBVHNodes = static_cast<int>(bvhNodes.size());
    host.bvhNodes = bvhNodes.empty() ? nullptr : bvhNodes.data();
    host.numBVHTriangles = static_cast<int>(bvhTriangles.size());
    host.bvhTriangles = bvhTriangles.empty() ? nullptr : bvhTriangles.data();
    host.bvhRootIndex = host.numBVHNodes > 0 ? 0 : -1;
    host.numAreaLights = static_cast<int>(areaLights.size());
    host.areaLights = areaLights.empty() ? nullptr : areaLights.data();
    host.numPointLights = static_cast<int>(pointLights.size());
    host.pointLights = pointLights.empty() ? nullptr : pointLights.data();
    host.numDirectionalLights = static_cast<int>(directionalLights.size());
    host.directionalLights = directionalLights.empty() ? nullptr : directionalLights.data();
    host.camera = camera;
    return host;
}
