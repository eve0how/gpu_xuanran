#!/usr/bin/env python3
"""Decimate dragon.obj (negative indices) to ~target_faces using quadric decimation."""
import argparse
import os
import sys


def load_obj_negative(path):
    """Load OBJ with negative vertex/normal indices; return vertices, faces as 0-based v indices."""
    verts = []
    norms = []
    faces = []  # list of (i0,i1,i2) vertex indices 0-based
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("vn "):
                p = line.split()
                norms.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                parts = line.split()[1:4]
                idx = []
                for vert in parts:
                    vtok = vert.split("/")[0]
                    vi = int(vtok)
                    if vi < 0:
                        vi = len(verts) + vi
                    else:
                        vi -= 1
                    idx.append(vi)
                faces.append(tuple(idx))
    return verts, norms, faces


def write_obj_positive(path, verts, faces, comment=""):
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for x, y, z in verts:
            f.write(f"v {x:.6g} {y:.6g} {z:.6g}\n")
        for i0, i1, i2 in faces:
            f.write(f"f {i0+1} {i1+1} {i2+1}\n")


def bounds(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def decimate_open3d(verts, faces, target_faces):
    import open3d as o3d
    import numpy as np

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    simplified = mesh.simplify_quadric_decimation(target_faces)
    simplified.remove_degenerate_triangles()
    out_v = np.asarray(simplified.vertices)
    out_f = np.asarray(simplified.triangles)
    return [tuple(row) for row in out_v], [tuple(row) for row in out_f]


def decimate_uniform(verts, faces, target_faces):
    """Fallback: keep every k-th face (rough decimation)."""
    step = max(1, len(faces) // target_faces)
    kept = faces[::step][:target_faces]
    used = sorted(set(i for tri in kept for i in tri))
    remap = {old: new for new, old in enumerate(used)}
    new_verts = [verts[i] for i in used]
    new_faces = [tuple(remap[i] for i in tri) for tri in kept]
    return new_verts, new_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", default="mesh/dragon.obj", nargs="?")
    ap.add_argument("output", default="mesh/dragon_simple.obj", nargs="?")
    ap.add_argument("--target", type=int, default=3500)
    args = ap.parse_args()

    verts, norms, faces = load_obj_negative(args.input)
    print(f"input: {len(verts)} verts, {len(faces)} faces", file=sys.stderr)
    bx, by, bz = bounds(verts)
    print(f"bounds x{bx} y{by} z{bz}", file=sys.stderr)

    try:
        out_v, out_f = decimate_open3d(verts, faces, args.target)
        method = "open3d quadric"
    except Exception as e:
        print(f"open3d failed ({e}), using uniform subsample", file=sys.stderr)
        out_v, out_f = decimate_uniform(verts, faces, args.target)
        method = "uniform subsample"

    comment = f"decimated from {os.path.basename(args.input)} via {method}; {len(out_f)} faces"
    write_obj_positive(args.output, out_v, out_f, comment)
    print(f"Wrote {args.output}: {len(out_v)} verts, {len(out_f)} faces ({method})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
