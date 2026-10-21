"""r4_topo.py DISPLAY_DIR OUT_JSONL [--sample] -- D145 reading 4 (2, 3): our display meshes (r4_mesh.py's *_mesh.npz)
read by the baselines' own mesh readers, and sampled by the baselines' sampler (CPU; B11's venv: libigl with CGAL,
trimesh; run through baselines/b12/scripts/b12_cpu.sh's environment).

Per mesh, as B12's b12_frames.py check(): b11_common.topo (closed, Euler characteristic, boundary and non-manifold edges,
watertight, winding) and components (faces joined through manifold edges, largest volume first) with each component's
genus, the self-intersecting face pairs by libigl's copyleft CGAL remesh_self_intersections (detect only), the
enclosed volume. --sample: the end states (RAW `end`) sampled at 300k by the pipeline's sample_volume_stratified (seed
97), as B12 / B14 sampled the baselines' meshes, into DISPLAY_DIR/MESH_LABEL_sample.npz (frames (1, N, 3), raws, tgt =
the run's own target sample, deliver_n) for ext_end145.py."""
import glob
import json
import os
import sys
import time

import numpy as np
import trimesh

from b11_common import components, topo

d, out = sys.argv[1], sys.argv[2]
do_sample = "--sample" in sys.argv
for path in sorted(glob.glob(f"{d}/*_mesh.npz")):
    t0 = time.time()
    tag = os.path.basename(path)[:-len("_mesh.npz")]
    z = np.load(path)
    V, F = np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)
    t = topo(V, F)
    comps = components(V, F)
    cg = []
    for Fc in comps:
        tc = topo(V, Fc)
        cg.append(dict(F=tc["F"], closed=tc["closed"], genus=tc["genus"], volume=tc["signed_volume"]))
    m = trimesh.Trimesh(V, F, process=False)
    try:
        import igl.copyleft.cgal as cgal
        _, _, IF, _, _ = cgal.remesh_self_intersections(V, F, detect_only=True, cutoff=2 ** 31 - 1)
        nsi = int(len(IF))
    except Exception as e:  # noqa: BLE001
        nsi = f"failed: {e}"
    closed_g = [c["genus"] for c in cg if c["genus"] is not None]
    row = dict(tag=tag, V=t["V"], F=t["F"], euler=t["euler"], closed=t["closed"], boundary_edges=t["boundary_edges"],
               nonmanifold_edges=t["nonmanifold_edges"], nonmanifold_vertices=t["nonmanifold_vertices"],
               watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent), components=len(cg),
               genus_largest=cg[0]["genus"] if cg else None, genus_sum_closed=int(sum(closed_g)),
               components_open=int(sum(1 for c in cg if not c["closed"])),
               small_components=[c["F"] for c in cg[1:21]], self_intersecting_pairs=nsi, volume=float(m.volume),
               t_check=round(time.time() - t0, 1))
    if do_sample and tag.endswith(("_end", "_ind", "_tgt")):
        from physmorph.sampling.mesh import sample_volume_stratified
        t1 = time.time()
        try:
            x = sample_volume_stratified(m, 300000, seed=97).astype(np.float32)
            stem = tag[:-len("_end")] if tag.endswith("_end") else tag
            np.savez(f"{d}/{stem}_sample.npz", frames=x[None], raws=np.array([0]), tgt=z["tgt"],
                     deliver_n=np.int64(1))
            row["sampled"] = True
        except Exception as e:  # noqa: BLE001
            row["sampled"] = f"failed: {e}"
        row["t_sample"] = round(time.time() - t1, 1)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
