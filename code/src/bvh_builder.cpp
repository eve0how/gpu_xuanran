// 文件说明：CPU 端 SAH 风格 BVH 构建，将三角形索引重排后扁平化为 GPU 节点数组。
// 原创性声明：参考已有代码（PBRT/GAMES101 BVH 思路与课件），
// 节点布局与 flatten 流程按本作业 GpuBVHNode 格式独立实现。

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

static AABB boundsOfTriangle(const GpuTriangle &tri) {
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

    AABB computeBoundsRange(int start, int count) const {
        AABB box = emptyAABB();
        for (int triIdx = 0; triIdx < count; ++triIdx) {
            mergeAABB(box, boundsOfTriangle(tris[indices[start + triIdx]]));
        }
        return box;
    }

    int buildRecursive(int start, int count) {
        const int nodeId = static_cast<int>(buildNodes.size());
        buildNodes.push_back({});
        buildNodes[nodeId].start = start;
        buildNodes[nodeId].count = count;
        buildNodes[nodeId].bbox = computeBoundsRange(start, count);

        if (count <= kMaxLeafTris) {
            return nodeId;
        }

        AABB centroidBounds = emptyAABB();
        for (int triIdx = 0; triIdx < count; ++triIdx) {
            const GpuTriangle &triPrim = tris[indices[start + triIdx]];
            expandAABB(centroidBounds, centroidAxis(triPrim, 0), centroidAxis(triPrim, 1),
                      centroidAxis(triPrim, 2));
        }

        int splitAxis = 0;
        float spanX = centroidBounds.bmax[0] - centroidBounds.bmin[0];
        float spanY = centroidBounds.bmax[1] - centroidBounds.bmin[1];
        float spanZ = centroidBounds.bmax[2] - centroidBounds.bmin[2];
        if (spanY > spanX && spanY > spanZ) {
            splitAxis = 1;
        } else if (spanZ > spanX && spanZ > spanY) {
            splitAxis = 2;
        }

        const int mid = start + count / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + start + count,
                         [this, splitAxis](int a, int b) {
                             return centroidAxis(tris[a], splitAxis) < centroidAxis(tris[b], splitAxis);
                         });

        const int leftId = buildRecursive(start, mid - start);
        buildNodes[nodeId].left = leftId;
        const int rightId = buildRecursive(mid, count - (mid - start));
        buildNodes[nodeId].right = rightId;
        return nodeId;
    }

    void flatten(int buildNodeId, std::vector<GpuBVHNode> &outNodes, std::vector<GpuTriangle> &outTriangles,
                 int &triOffset) {
        const BuildNode &nodeRec = buildNodes[buildNodeId];
        const int nodeIdx = static_cast<int>(outNodes.size());
        outNodes.push_back({});

        for (int axis = 0; axis < 3; ++axis) {
            outNodes[nodeIdx].bboxMin[axis] = nodeRec.bbox.bmin[axis];
            outNodes[nodeIdx].bboxMax[axis] = nodeRec.bbox.bmax[axis];
        }

        if (nodeRec.isLeaf()) {
            outNodes[nodeIdx].leftChild = triOffset;
            outNodes[nodeIdx].rightChild = 0;
            outNodes[nodeIdx].primitiveCount = nodeRec.count;
            outNodes[nodeIdx]._pad = 0;
            for (int leafTri = 0; leafTri < nodeRec.count; ++leafTri) {
                outTriangles.push_back(tris[indices[nodeRec.start + leafTri]]);
            }
            triOffset += nodeRec.count;
            return;
        }

        outNodes[nodeIdx].primitiveCount = 0;
        outNodes[nodeIdx]._pad = 0;
        outNodes[nodeIdx].leftChild = static_cast<int>(outNodes.size());
        flatten(nodeRec.left, outNodes, outTriangles, triOffset);
        outNodes[nodeIdx].rightChild = static_cast<int>(outNodes.size());
        flatten(nodeRec.right, outNodes, outTriangles, triOffset);
    }
};

}  // namespace

void buildBVH(const std::vector<GpuTriangle> &triangles, std::vector<GpuBVHNode> &outNodes,
              std::vector<GpuTriangle> &outTriangles) {
    BVHBuilder builder(triangles);
    builder.build(outNodes, outTriangles);
}
