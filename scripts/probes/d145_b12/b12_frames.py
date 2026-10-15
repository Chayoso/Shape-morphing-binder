"""B12 (D145) CPU readout of one run, in B11's venv (b12_cpu.sh, B11's runner with B12's temp dir: libigl with its copyleft CGAL
module, trimesh, repo_d145's sampler on PYTHONPATH; no card).

Per frame mesh (results/<method>/<tag>/frames.npz, 21 frames at matched progress of the method's own loss):
  topology   closedness, Euler characteristic, components (B11's b11_common.components: faces joined through
             manifold edges) and the genus of each closed component, their sum and the largest component's;
  watertight trimesh's is_watertight;
  self-intersection  self-intersecting face pairs by libigl's copyleft CGAL remesh_self_intersections (detect only),
             as B11;
  volume     the enclosed signed volume against the target mesh's (B12's target, B5's export, holes fan-filled) and
             the target's fill volume (B5's export.json);
then every frame sampled at 300k by the pipeline's sample_volume_stratified(mesh, 300000, seed=97) into
<OUT>/<name>300k_<LABEL>_frames12.npz (B5 / B11's format: frames, raws, tgt = the run's own target sample, deliver_n)
for B11's end check, D145's volume routine and its harness.

  b12_cpu.sh b12_frames.py RUNDIR NAME LABEL OUTDIR
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import trimesh

from b11_common import components, topo

RUN, NAME, LABEL, OUTDIR = sys.argv[1:5]
B5 = "/data/relcfd/chayo/physmorph_v2/baselines/buonomo/inputs"
N, SEED = 300000, 97
z = dict(np.load(f"{RUN}/frames.npz"))          # read eagerly: forked workers must not share one zip handle
K = len(z["raws"])
raws = np.asarray(z["raws"]).astype(np.int64)
exp = json.load(open(f"{B5}/{NAME}/export.json"))
run = json.load(open(f"{RUN}/run.json"))
vol_tgt = float(run["target"]["volume"])


def frame(k):
    return np.asarray(z[f"v{k:02d}"], np.float64), np.asarray(z[f"f{k:02d}"], np.int64)


def check(k):
    import igl.copyleft.cgal as cgal
    t0 = time.time()
    V, F = frame(k)
    if len(F) == 0:
        return dict(k=k, raw=int(raws[k]), empty=True)
    t = topo(V, F)
    comps = components(V, F)
    cg = []
    for Fc in comps:
        tc = topo(V, Fc)
        cg.append(dict(F=tc["F"], closed=tc["closed"], euler=tc["euler"], genus=tc["genus"],
                       volume=tc["signed_volume"]))
    m = trimesh.Trimesh(V, F, process=False)
    try:
        _, _, IF, _, _ = cgal.remesh_self_intersections(V, F, detect_only=True, cutoff=2 ** 31 - 1)
        nsi = int(len(IF))
        nsi_f = int(len(np.unique(IF))) if nsi else 0
    except Exception as e:  # noqa: BLE001
        nsi, nsi_f = f"failed: {e}", None
    vol = float(m.volume)
    closed_g = [c["genus"] for c in cg if c["genus"] is not None]
    return dict(k=k, raw=int(raws[k]), q=float(z["q"][k]), V=t["V"], F=t["F"], euler=t["euler"], closed=t["closed"],
                boundary_edges=t["boundary_edges"], nonmanifold_edges=t["nonmanifold_edges"],
                nonmanifold_vertices=t["nonmanifold_vertices"], watertight=bool(m.is_watertight),
                winding_consistent=bool(m.is_winding_consistent), components=len(cg),
                genus_largest=cg[0]["genus"] if cg else None, genus_sum_closed=int(sum(closed_g)),
                components_open=int(sum(1 for c in cg if not c["closed"])), largest_volume=cg[0]["volume"] if cg else None,
                self_intersecting_pairs=nsi, self_intersecting_faces=nsi_f, volume=vol,
                vol_ratio_tgt_mesh=vol / vol_tgt, vol_ratio_tgt_fill=vol / exp["fill_tgt"], seconds=time.time() - t0)


def sample(k):
    from physmorph.sampling.mesh import sample_volume_stratified
    t0 = time.time()
    V, F = frame(k)
    try:
        x = sample_volume_stratified(trimesh.Trimesh(V, F, process=False), N, seed=SEED).astype(np.float32)
        return k, x, None, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return k, None, repr(e), time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    pool_n = int(os.environ.get("B12_POOL", "6"))
    with Pool(pool_n) as pool:
        rows = pool.map(check, range(K))
    t_check = time.time() - t0
    with Pool(pool_n) as pool:
        res = pool.map(sample, range(K))
    ok = [r for r in res if r[1] is not None]
    fails = [dict(k=r[0], error=r[2]) for r in res if r[1] is None]
    os.makedirs(OUTDIR, exist_ok=True)
    npz = f"{OUTDIR}/{NAME}300k_{LABEL}_frames12.npz"
    if ok:
        np.savez(npz, frames=np.stack([r[1] for r in ok]).astype(np.float32), raws=raws[[r[0] for r in ok]],
                 tgt=np.asarray(np.load(f"{B5}/{NAME}/pipeline_samples.npz")["tgt"], np.float32),
                 deliver_n=np.int64(len(ok)), q=np.asarray(z["q"])[[r[0] for r in ok]])
    out = dict(run=RUN, name=NAME, label=LABEL, npz=npz if ok else None, frames=rows, sampled=len(ok), failed=fails,
               seconds=dict(check=t_check, sample=time.time() - t0 - t_check),
               per_frame_sample_s=[float(r[3]) for r in res], tgt_mesh_volume=vol_tgt, tgt_fill_volume=exp["fill_tgt"])
    with open(f"{RUN}/frames_check.json", "w") as fh:
        json.dump(out, fh, indent=1)
    end = rows[-1]
    print(f"[{LABEL}] {NAME}: {len(ok)} / {K} frames sampled ({len(fails)} failed) in {time.time() - t0:.0f} s; end: "
          f"genus largest {end.get('genus_largest')}, components {end.get('components')}, watertight "
          f"{end.get('watertight')}, SI pairs {end.get('self_intersecting_pairs')}, vol/tgt "
          f"{end.get('vol_ratio_tgt_mesh', float('nan')):.4f}", flush=True)
