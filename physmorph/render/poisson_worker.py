"""Screened Poisson reconstruction in an ISOLATED interpreter.

Open3D 0.19's create_from_point_cloud_poisson segfaults now and then when called frame after
frame in one process (dragon 150k video: exit 139 at frame 126 of 327, then at frame 51 on the
retry — not a frame, a race inside the reconstruction). A segfault in a child cannot take the
render down: the parent gets no result, retries once single-threaded, and otherwise reports
None so the caller can fall back for that frame and record it.

The child is a plain `python -m physmorph.render.poisson_worker in.npz out.npz` — not a
multiprocessing spawn, which re-executes the parent's main module (the renderer script is
not import-safe: it parses arguments and loads the archive at import).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np


def _reconstruct(points, normals, depth, n_threads):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), linear_fit=False, n_threads=int(n_threads))
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)


def poisson_isolated(points: np.ndarray, normals: np.ndarray, depth: int, n_threads: int = -1,
                     retries: int = 1, timeout: float = 1800.0):
    """(vertices float32 (V,3), triangles int32 (F,3)) or None if the reconstruction failed
    `retries + 1` times (the retry runs single-threaded)."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ); env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory(prefix="poisson_") as td:
        fin, fout = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        for attempt in range(retries + 1):
            # the thread cap (2026-09-22): -1 = every core, and five concurrent frames of the gallery videos put a
            # 128-core host at load 290 (other users' jobs included); PHYSMORPH_POISSON_THREADS caps each child
            cap = int(os.environ.get("PHYSMORPH_POISSON_THREADS", "16"))
            nt = (cap if n_threads < 0 or n_threads > cap else n_threads) if attempt == 0 else 1
            np.savez(fin, points=points.astype(np.float32), normals=normals.astype(np.float32),
                     depth=np.int64(depth), n_threads=np.int64(nt))
            if os.path.exists(fout):
                os.remove(fout)
            try:
                r = subprocess.run([sys.executable, "-m", "physmorph.render.poisson_worker", fin, fout],
                                   env=env, cwd=repo, timeout=timeout, capture_output=True, text=True)
                code = r.returncode
                err = (r.stderr or "").strip().splitlines()[-1:] if code != 0 else []
            except subprocess.TimeoutExpired:
                code, err = "timeout", []
            if code == 0 and os.path.exists(fout):
                z = np.load(fout)
                return z["v"], z["f"]
            print(f"[poisson_worker] attempt {attempt + 1} failed (exit {code}, threads {nt}) {err}", flush=True)
    return None


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    v, f = _reconstruct(z["points"], z["normals"], int(z["depth"]), int(z["n_threads"]))
    np.savez(sys.argv[2], v=v, f=f)
