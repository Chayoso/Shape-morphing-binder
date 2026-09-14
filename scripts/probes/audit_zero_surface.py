"""Server-only rest-state diagnostic for material surface intersection reports."""
import argparse
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.pipeline.geometric import forward_geometry, trajectory_health, GeometricConfig
from physmorph.pipeline.surface_validity import surface_intersection_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("rollout diagnostics run on hyde06")
    folder = Path(args.fixture)
    meta = json.loads((folder / "metadata.json").read_text())
    with np.load(folder / "fixture.npz") as a:
        data = {k: a[k] for k in a.files}
    spec = RolloutSpec(data["src"], meta["particle_mass"], meta["lam"], meta["mu"],
        MPMParams(**meta["discretization"]), 64, device="cuda", vol0=data["vol0"],
        surface0=data["surface0"], surface_faces=data["surface_faces"])
    _, tr = forward_geometry(torch.zeros((64, len(spec.x0), 3, 3), device="cuda"), spec)
    frames = np.stack([a.numpy() for a in tr.surface_x])
    records = []
    for i, x in enumerate(frames):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x),
            o3d.utility.Vector3iVector(spec.surface_faces))
        pairs, candidates = surface_intersection_pairs(x, spec.surface_faces)
        records.append({"frame": i, "pairs": pairs.tolist(),
            "certified_disjoint_candidates": candidates-len(pairs),
            "max_move": float(np.linalg.norm(x - frames[0], axis=1).max())})
        if len(pairs):
            faces = spec.surface_faces[pairs[:2]]
            records[-1].update(first_faces=faces.tolist(), first_triangles=x[faces].tolist())
    report = {"discretization": meta["discretization"], "N": len(spec.x0),
        "centers": {k: data[k].mean(0, dtype=np.float64).tolist()
            for k in ("src", "tgt", "dense_target")},
        "local_health": trajectory_health(tr, GeometricConfig()), "frames": records}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}))
    print(json.dumps([r for r in records if r["pairs"]][:2]))


if __name__ == "__main__":
    main()
