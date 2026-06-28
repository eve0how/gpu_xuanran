#include "bvh_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int kMaxLeafTris = 4;

static AABB emptyAABB() {
    AABB b{};
    b.bmin[0] = b.bmin[1] = b.bmin[2] = std::numeric_limits<float>::max();
    b.bmax[0] = b.bmax[1] = b.bmax[2] = -std::numeric_limits<float>::max();
    return b;
}

static void expandAABB(AABB &box, float x, float y, float z) {
    box.bmin[0] = std::min(box.bmin[0], x);
    box.bmin[1] = std::min(box.bmin[1], y);
    box.bmin[2] = std::min(box.bmin[2], z);
    box.bmax[0] = std::max(box.bmax[0], x);
    box.bmax[1] = std::max(box.bmax[1], y);
    box.bmax[2] = std::max(box.bmax[2], z);
}

static AABB triangleAABB(const GpuTriangle &tri) {
    AABB box = emptyAABB();
    expandAABB(box, tri.v0[0], tri.v0[1], tri.v0[2]);
    expandAABB(box, tri.v1[0], tri.v1[1], tri.v1[2]);
    expandAABB(box, tri.v2[0], tri.v2[1], tri.v2[2]);
    return box;
}

static void mergeAABB(AABB &dst, const AABB &src) {
    for (int i = 0; i < 3; ++i) {
        dst.bmin[i] = std::min(dst.bmin[i], src.bmin[i]);
        dst.bmax[i] = std::max(dst.bmax[i], src.bmax[i]);
    }
}

static float centroidAxis(const GpuTriangle &tri, int axis) {
    float c0 = tri.v0[axis];
    float c1 = tri.v1[axis];
    float c2 = tri.v2[axis];
    return (c0 + c1 + c2) / 3.0f;
}

struct BuildNode {
    AABB bbox{};
    int start = 0;
    int count = 0;
    int left = -1;
    int right = -1;
    bool isLeaf() const { return left < 0; }
};

class BVHBuilder {
public:
    explicit BVHBuilder(const std::vector<GpuTriangle> &triangles)
        : tris(triangles) {
        indices.resize(tris.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = static_cast<int>(i);
        }
    }

    void build(std::vector<GpuBVHNode> &outNodes, std::vector<GpuTriangle> &outTriangles) {
        if (tris.empty()) {
            outNodes.clear();
            outTriangles.clear();
            return;
        }
        buildNodes.clear();
        buildRecursive(0, static_cast<int>(tris.size()));
        outNodes.clear();
        outTriangles.clear();
        int triOffset = 0;
        flatten(0, outNodes, outTriangles, triOffset);
    }

private:
    const std::vector<GpuTriangle> &tris;
    std::vector<int> indices;
    std::vector<BuildNode> buildNodes;

    AABB bounds(int start, int count) const {
        AABB box = emptyAABB();
        for (int i = 0; i < count; ++i) {
            mergeAABB(box, triangleAABB(tris[indices[start + i]]));
        }
        return box;
    }

    int buildRecursive(int start, int count) {
        const int nodeId = static_cast<int>(buildNodes.size());
        buildNodes.push_back({});
        buildNodes[nodeId].start = start;
        buildNodes[nodeId].count = count;
        buildNodes[nodeId].bbox = bounds(start, count);

        if (count <= kMaxLeafTris) {
            return nodeId;
        }

        AABB centroidBounds = emptyAABB();
        for (int i = 0; i < count; ++i) {
            const GpuTriangle &tri = tris[indices[start + i]];
            expandAABB(centroidBounds, centroidAxis(tri, 0), centroidAxis(tri, 1), centroidAxis(tri, 2));
        }

        int axis = 0;
        float extentX = centroidBounds.bmax[0] - centroidBounds.bmin[0];
        float extentY = centroidBounds.bmax[1] - centroidBounds.bmin[1];
        float extentZ = centroidBounds.bmax[2] - centroidBounds.bmin[2];
        if (extentY > extentX && extentY > extentZ) {
            axis = 1;
        } else if (extentZ > extentX && extentZ > extentY) {
            axis = 2;
        }

        const int mid = start + count / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + start + count,
                         [this, axis](int a, int b) {
                             return centroidAxis(tris[a], axis) < centroidAxis(tris[b], axis);
                         });

        const int leftId = buildRecursive(start, mid - start);
        buildNodes[nodeId].left = leftId;
        const int rightId = buildRecursive(mid, count - (mid - start));
        buildNodes[nodeId].right = rightId;
        return nodeId;
    }

    void flatten(int buildNodeId, std::vector<GpuBVHNode> &outNodes, std::vector<GpuTriangle> &outTriangles,
                 int &triOffset) {
        const BuildNode &bn = buildNodes[buildNodeId];
        const int nodeIdx = static_cast<int>(outNodes.size());
        outNodes.push_back({});

        for (int i = 0; i < 3; ++i) {
            outNodes[nodeIdx].bboxMin[i] = bn.bbox.bmin[i];
            outNodes[nodeIdx].bboxMax[i] = bn.bbox.bmax[i];
        }

        if (bn.isLeaf()) {
            outNodes[nodeIdx].leftChild = triOffset;
            outNodes[nodeIdx].rightChild = 0;
            outNodes[nodeIdx].primitiveCount = bn.count;
            outNodes[nodeIdx]._pad = 0;
            for (int i = 0; i < bn.count; ++i) {
                outTriangles.push_back(tris[indices[bn.start + i]]);
            }
            triOffset += bn.count;
            return;
        }

        outNodes[nodeIdx].primitiveCount = 0;
        outNodes[nodeIdx]._pad = 0;
        outNodes[nodeIdx].leftChild = static_cast<int>(outNodes.size());
        flatten(bn.left, outNodes, outTriangles, triOffset);
        outNodes[nodeIdx].rightChild = static_cast<int>(outNodes.size());
        flatten(bn.right, outNodes, outTriangles, triOffset);
    }
};

}  // namespace

void buildBVH(const std::vector<GpuTriangle> &triangles, std::vector<GpuBVHNode> &outNodes,
              std::vector<GpuTriangle> &outTriangles) {
    BVHBuilder builder(triangles);
    builder.build(outNodes, outTriangles);
}
