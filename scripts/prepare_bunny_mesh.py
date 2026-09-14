"""Repair a derived bunny asset and independently validate it; leave OBJ unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import socket

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("asset preparation runs on hyde06")
    import trimesh
    import open3d as o3d
    import pymeshfix
    out = Path(args.out)
    if out.exists():
        raise ValueError("derived-asset directory must be new")
    out.mkdir(parents=True)
    source = trimesh.load(args.mesh, force="mesh", process=True)
    v, f = pymeshfix.clean_from_arrays(np.asarray(source.vertices, np.float64),
        np.asarray(source.faces, np.int32), verbose=False, joincomp=False, remove_smallest_components=False)
    # Validate the actual float32 asset used by MPM/rendering, not only a float64 intermediate.
    fixed = trimesh.Trimesh(np.asarray(v, np.float32), f, process=True, validate=True)
    if fixed.volume < 0:
        fixed.invert()
    legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(fixed.vertices),
                                     o3d.utility.Vector3iVector(fixed.faces))
    report = {"repair_package": "pymeshfix", "repair_version": pymeshfix.__version__,
        "input_sha256": hashlib.sha256(Path(args.mesh).read_bytes()).hexdigest(),
        "original_faces": len(source.faces), "original_vertices": len(source.vertices),
        "faces": len(fixed.faces), "vertices": len(fixed.vertices), "volume": float(fixed.volume),
        "watertight": bool(fixed.is_watertight), "winding": bool(fixed.is_winding_consistent),
        "euler": int(fixed.euler_number), "components": len(fixed.split(only_watertight=False)),
        "self_intersection_pairs": len(legacy.get_self_intersecting_triangles()),
        "min_triangle_area": float(fixed.area_faces.min()),
        "repair_options": {"joincomp": False, "remove_smallest_components": False}}
    np.savez_compressed(out/"mesh.npz", vertices=np.asarray(fixed.vertices, np.float32), faces=np.asarray(fixed.faces, np.int64))
    report["output_sha256"] = hashlib.sha256((out/"mesh.npz").read_bytes()).hexdigest()
    report["valid"] = (report["watertight"] and report["winding"] and report["euler"] == 2
        and report["components"] == 1 and report["self_intersection_pairs"] == 0 and report["min_triangle_area"] > 0)
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if not report["valid"]:
        raise RuntimeError("derived asset failed independent validation; retained for diagnosis")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
