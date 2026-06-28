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

static void toGpuVec3(const Vector3f &v, float out[3]) {
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
}

static Vector3f transformPointMat(const Matrix4f &m, const Vector3f &p) {
    return (m * Vector4f(p, 1.0f)).xyz();
}

static float uniformScaleFactor(const Matrix4f &m) {
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

    void expandBbox(const Vector3f &p) {
        hasBbox = true;
        bboxMin.x() = std::min(bboxMin.x(), p.x());
        bboxMin.y() = std::min(bboxMin.y(), p.y());
        bboxMin.z() = std::min(bboxMin.z(), p.z());
        bboxMax.x() = std::max(bboxMax.x(), p.x());
        bboxMax.y() = std::max(bboxMax.y(), p.y());
        bboxMax.z() = std::max(bboxMax.z(), p.z());
    }

    void expandBboxTriangle(const Vector3f &a, const Vector3f &b, const Vector3f &c) {
        expandBbox(a);
        expandBbox(b);
        expandBbox(c);
    }

    void buildMaterialMap() {
        for (int i = 0; i < scene.getNumMaterials(); ++i) {
            matMap[scene.getMaterial(i)] = i;
        }
    }

    int matIdFor(Material *mat) const {
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
            toGpuVec3(mat->getDiffuseColor(), g.diffuse);
            g.emission[0] = g.emission[1] = g.emission[2] = 0.0f;
            g.specular[0] = g.specular[1] = g.specular[2] = 0.0f;
            g.ior = 1.0f;
            g.roughness = 0.2f;
            g.f0[0] = g.f0[1] = g.f0[2] = 0.04f;
            g.dispersionDelta = 0.0f;
            g.shininess = 0.0f;
            g.fresnelEnabled = 1;
            g.alphaX = 0.2f;
            g.alphaY = 0.2f;
            g.tangent[0] = 1.0f;
            g.tangent[1] = 0.0f;
            g.tangent[2] = 0.0f;
            toGpuVec3(mat->getSpecularColor(), g.specular);
            g.shininess = mat->getShininess();

            switch (mat->getType()) {
                case MaterialType::REFLECT: {
                    auto *rm = static_cast<ReflectMaterial *>(mat);
                    toGpuVec3(rm->getReflectColor(), g.specular);
                    break;
                }
                case MaterialType::REFRACT: {
                    auto *rm = static_cast<RefractMaterial *>(mat);
                    toGpuVec3(rm->getRefractColor(), g.specular);
                    g.ior = rm->getRefractIndex();
                    g.dispersionDelta = rm->getDispersionDelta();
                    g.fresnelEnabled = rm->isFresnelEnabled() ? 1 : 0;
                    break;
                }
                case MaterialType::EMISSIVE:
                    toGpuVec3(mat->getEmission(), g.emission);
                    break;
                case MaterialType::GLOSSY: {
                    auto *gm = static_cast<GlossyMaterial *>(mat);
                    toGpuVec3(gm->getSpecularColor(), g.specular);
                    g.roughness = gm->getRoughness();
                    toGpuVec3(gm->getF0(), g.f0);
                    break;
                }
                case MaterialType::WARD: {
                    auto *wm = static_cast<WardMaterial *>(mat);
                    toGpuVec3(wm->getSpecularColor(), g.specular);
                    g.alphaX = wm->getAlphaX();
                    g.alphaY = wm->getAlphaY();
                    toGpuVec3(wm->getTangentDirection(), g.tangent);
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
                toGpuVec3(area->getVertex0(), g.v0);
                toGpuVec3(area->getVertex1(), g.v1);
                toGpuVec3(area->getVertex2(), g.v2);
                toGpuVec3(area->getColor(), g.color);
                areaLights->push_back(g);
                expandBboxTriangle(area->getVertex0(), area->getVertex1(), area->getVertex2());
            } else if (auto *point = dynamic_cast<PointLight *>(scene.getLight(i))) {
                GpuPointLight g{};
                toGpuVec3(point->getPosition(), g.pos);
                Vector3f dir, col;
                point->getIllumination(Vector3f::ZERO, dir, col);
                toGpuVec3(col, g.color);
                pointLights->push_back(g);
            } else if (auto *dirLight = dynamic_cast<DirectionalLight *>(scene.getLight(i))) {
                GpuDirectionalLight g{};
                Vector3f L, col;
                dirLight->getIllumination(Vector3f::ZERO, L, col);
                toGpuVec3(L, g.direction);
                toGpuVec3(col, g.color);
                directionalLights->push_back(g);
            }
        }
    }

    void buildCamera() {
        Camera *cam = scene.getCamera();
        toGpuVec3(cam->getCenter(), camera->center);
        toGpuVec3(cam->getDirection(), camera->direction);
        toGpuVec3(cam->getHorizontal(), camera->horizontal);
        toGpuVec3(cam->getUp(), camera->up);
        toGpuVec3(scene.getBackgroundColor(), camera->bg);
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
        toGpuVec3(a, tri.v0);
        toGpuVec3(b, tri.v1);
        toGpuVec3(c, tri.v2);
        tri.matId = matIdFor(mat);
        triangles->push_back(tri);
    }

    void flattenObject(Object3D *obj, const Matrix4f &world) {
        if (obj == nullptr) {
            return;
        }
        if (auto *group = dynamic_cast<Group *>(obj)) {
            for (int i = 0; i < group->getGroupSize(); ++i) {
                flattenObject(group->getObject(i), world);
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
            toGpuVec3(c, g.center);
            g.radius = sphere->getRadius() * uniformScaleFactor(world);
            g.matId = matIdFor(sphere->getMaterial());
            spheres->push_back(g);
            float r = g.radius;
            expandBbox(c + Vector3f(r, r, r));
            expandBbox(c - Vector3f(r, r, r));
            return;
        }
        if (auto *plane = dynamic_cast<Plane *>(obj)) {
            GpuPlane g{};
            Vector3f n = (world * Vector4f(plane->getNormal(), 0.0f)).xyz().normalized();
            Vector3f p = transformPointMat(world, plane->getNormal() * plane->getOffset());
            toGpuVec3(n, g.normal);
            g.offset = Vector3f::dot(n, p);
            g.matId = matIdFor(plane->getMaterial());
            planes->push_back(g);
            // Large finite slab for plane bbox (Cornell walls).
            Vector3f absN(fabsf(n.x()), fabsf(n.y()), fabsf(n.z()));
            if (absN.y() > 0.9f) {
                expandBbox(Vector3f(-2.0f, p.y(), -2.0f));
                expandBbox(Vector3f(2.0f, p.y(), 2.0f));
            } else if (absN.x() > 0.9f) {
                expandBbox(Vector3f(p.x(), -2.0f, -2.0f));
                expandBbox(Vector3f(p.x(), 2.0f, 2.0f));
            } else {
                expandBbox(Vector3f(-2.0f, -2.0f, p.z()));
                expandBbox(Vector3f(2.0f, 2.0f, p.z()));
            }
            return;
        }
        if (auto *tri = dynamic_cast<Triangle *>(obj)) {
            Vector3f a = transformPointMat(world, tri->vertices[0]);
            Vector3f b = transformPointMat(world, tri->vertices[1]);
            Vector3f c = transformPointMat(world, tri->vertices[2]);
            addTriangle(a, b, c, tri->getMaterial());
            expandBboxTriangle(a, b, c);
            return;
        }
        if (auto *mesh = dynamic_cast<Mesh *>(obj)) {
            for (const auto &idx : mesh->t) {
                Vector3f a = transformPointMat(world, mesh->v[idx.x[0]]);
                Vector3f b = transformPointMat(world, mesh->v[idx.x[1]]);
                Vector3f c = transformPointMat(world, mesh->v[idx.x[2]]);
                addTriangle(a, b, c, mesh->getMaterial());
                expandBboxTriangle(a, b, c);
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
