"""Server-only raw trajectory audit; never reads an image or a rendering operator."""
import argparse
import json
from pathlib import Path
import socket
import sys

import numpy as np
import open3d as o3d
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.pipeline.surface_validity import surface_intersection_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("trajectory audits run on hyde06")
    run = Path(args.run)
    meta = json.loads((run/"metadata.json").read_text())
    with np.load(run/"trajectory.npz") as a:
        particles, surface, faces = a["frames"], a["surface_frames"], a["surface_faces"]
    prm = meta["fixture"]["discretization"]
    tolerance = 1e-3*prm["dx"]
    records = []
    for index, (x, sx) in enumerate(zip(particles, surface)):
        legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(sx), o3d.utility.Vector3iVector(faces))
        intersections, candidates = surface_intersection_pairs(sx, faces)
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        signed = scene.compute_signed_distance(o3d.core.Tensor(np.ascontiguousarray(x, np.float32)),
                                                nthreads=4, nsamples=3).numpy()
        # Signed distance assumes a closed nonintersecting mesh. If that condition
        # fails, retain diagnostics but never call their sign a containment proof.
        records.append({"frame": index, "time": index*prm["dt"],
            "self_intersection_pairs": len(intersections),
            "certified_disjoint_candidates": candidates-len(intersections),
            "outside_fraction": float((signed > tolerance).mean()),
            "max_outside_distance": float(np.maximum(signed, 0).max()),
            "signed_distance_valid": len(intersections) == 0 and trimesh.Trimesh(sx, faces, process=False).is_watertight})
    result = {"discretization": prm, "particles": len(particles[0]), "surface_vertices": len(surface[0]),
        "surface_faces": len(faces), "outside_tolerance": tolerance,
        "max_self_intersection_pairs": max(r["self_intersection_pairs"] for r in records),
        "max_outside_fraction": max(r["outside_fraction"] for r in records),
        "max_outside_distance": max(r["max_outside_distance"] for r in records),
        "all_signs_valid": all(r["signed_distance_valid"] for r in records), "frames": records}
    (run/"surface_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}), flush=True)


if __name__ == "__main__":
    main()
