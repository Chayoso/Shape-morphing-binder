"""Screened Poisson reconstruction in an ISOLATED child process.

Open3D 0.19's create_from_point_cloud_poisson segfaults now and then when called frame after
frame in one process (dragon 150k video: exit 139 at frame 126 of 327, then at frame 51 on the
retry — not a frame, a race inside the reconstruction). A segfault in a child cannot take the
render down: the parent gets no result, retries once single-threaded, and otherwise reports
None so the caller can fall back for that frame and record it.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _run(points, normals, depth, n_threads, conn):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), linear_fit=False, n_threads=int(n_threads))
    conn.send((np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)))
    conn.close()


def poisson_isolated(points: np.ndarray, normals: np.ndarray, depth: int, n_threads: int = -1,
                     retries: int = 1):
    """(vertices float32 (V,3), triangles int32 (F,3)) or None if the reconstruction crashed
    `retries + 1` times (the retry runs single-threaded)."""
    ctx = mp.get_context("spawn")
    for attempt in range(retries + 1):
        nt = n_threads if attempt == 0 else 1
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_run, args=(points, normals, depth, nt, child), daemon=True)
        p.start(); child.close()
        result = None
        try:
            if parent.poll(1800):
                result = parent.recv()
        except (EOFError, OSError):
            result = None
        p.join(30)
        if p.is_alive():
            p.kill(); p.join()
        parent.close()
        if result is not None and p.exitcode == 0:
            return result
        print(f"[poisson_worker] attempt {attempt + 1} failed (exit code {p.exitcode}, threads {nt})", flush=True)
    return None
