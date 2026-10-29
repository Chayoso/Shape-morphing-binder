"""r5_d157x.py OUT_JSONL JOB [JOB ...] | OUT_JSONL @JOBS_FILE -- D145 reading 5 (2) and (1): D157's reader on states it
was not given: the external methods' ends and the cell-size runs (CPU; run from tmp/d157 with its env.sh).

D157's code is imported unchanged from tmp/d157/d157_read.py. A JOB is "MESH ROUTE LABEL PATH":
  pts   PATH an archive (frames, tgt; its last finite frame is the state): the state's Zhu-Bridson exterior (zb: the
        silhouettes and the 3D reading) and its display (zbs: the held-out studio images), against D157's raw and
        corrected referees -- exactly D157's route for ours;
  mesh  PATH the method's own end mesh (.ply, or B12's frames.npz: its last vNN / fNN): the mesh itself as the surface
        (face normals for the silhouettes and the 3D reading, crease-aware corner normals for the images), against the
        same referees; also its vertices' distance to the raw mesh (pitches).
The context of each mesh (the referees, the pitch, the box, the thin set, the corrected offset c) is D157's Ctx built
from B1's L97 end (seed 97: the frame the external methods' ends and the cell-size runs share), so every row sits beside
D157's own rows for ours. Jobs of one mesh must be consecutive. Rows: D157's keys (sil / d3 / img per surface)."""
import json
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d157")
import d157_read as d  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"


def last_finite(path):
    z = np.load(path, allow_pickle=True)
    F = z["frames"]
    k = len(F) - 1
    while k > 0 and not np.isfinite(np.asarray(F[k][:16])).all():
        k -= 1
    return np.asarray(F[k], np.float32), int(k)


def native(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        k = len(z["raws"]) - 1
        return np.asarray(z[f"v{k:02d}"], np.float64), np.asarray(z[f"f{k:02d}"], np.int64)
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def read_surface(ctx, row, kind, st, st_img, masks, seen, rng, do_sil=True, do_img=True):
    a, ext = ctx.a, ctx.extent
    if do_sil:
        sil = {}
        for r in d.RES:
            Ms = [d.MaskInfo(mm) for mm in masks[r]]
            for rn in ("raw", "cor"):
                per = [d.sil_metrics(Ms[i], ctx.rmask[rn][r][i], r, ext, a) for i in range(len(d.VIEWS))]
                sil.setdefault(rn, {})[str(r)] = d.summarize_views(per)
            del Ms
        row["sil"][kind] = sil
        sP, sN, _ = d.area_sample(st, seen, d.NPTS, rng)
        d3 = {}
        for rn in ("raw", "cor"):
            rP, rN, thin = ctx.rsamp[rn]
            ref = ctx.raw_fn_only if rn == "raw" else ctx.grown
            d3[rn], _ = d.d3_metrics(st, sP, sN, ref, rP, rN, a, thin)
        row["d3"][kind] = d3
    if do_img:
        S = [d.render(st_img, ctx.center, ctx.radius, v) for v in d.HELD]
        row["img"][kind] = {rn: d.img_metrics(S, ctx.rimg[rn]) for rn in ("raw", "cor")}


def main():
    out = sys.argv[1]
    jobs = sys.argv[2:]
    if len(jobs) == 1 and jobs[0].startswith("@"):
        jobs = [ln.strip() for ln in open(jobs[0][1:]) if ln.strip() and not ln.startswith("#")]
    with open(out, "a") as fh:
        for job in jobs:
            mesh, route, label, path = job.split()
            t0 = time.time()
            ctx = d.Ctx.get(mesh, 97, "300k", f"{G}/d145t_b1/{mesh}300k_L97_end1.npz")
            row = dict(mesh=mesh, route=route, label=label, path=path, ctx=ctx.info, sil={}, d3={}, img={})
            code = {"L": 1, "P": 2, "V": 1, "W": 2, "own": 3}.get(label, 4)
            rng = np.random.default_rng(97 * 7 + code)
            try:
                if route == "pts":
                    x, k = last_finite(path)
                    row["frame"] = k
                    # a scattered end: d157_read's field mesh of Xu et al.'s homer XU101 (3.7 % of the particles outside
                    # the largest set linked at 1.5 pitches) grew past 80 GB of the server's memory; an end with more than
                    # 1 % apart is recorded and not read (Xu et al.'s homer and dragon, OT's dragon)
                    from scipy.sparse import coo_matrix
                    from scipy.sparse.csgraph import connected_components
                    from scipy.spatial import cKDTree
                    pr = cKDTree(x).query_pairs(1.5 * ctx.a, output_type="ndarray")
                    _, lab2 = connected_components(coo_matrix((np.ones(len(pr), np.int8), (pr[:, 0], pr[:, 1])),
                                                              shape=(len(x), len(x))), directed=False)
                    row["apart_share"] = float(1 - np.bincount(lab2).max() / len(x))
                    if row["apart_share"] > 0.01:
                        raise RuntimeError(f"scattered end: {row['apart_share']:.1%} apart; not read")
                    for kind in d.SURFS:                     # zb: silhouettes and 3D; zbs: the studio images
                        st, info = d.field_mesh(kind, x, ctx.a)
                        row[f"{kind}_mesh"] = info
                        masks, seen = d.silhouettes(st, ctx.extent) if kind == "zb" else (None, None)
                        read_surface(ctx, row, kind, st, st, masks, seen, rng, do_sil=(kind == "zb"),
                                     do_img=(kind == "zbs"))
                        del st
                else:
                    V, F = native(path)
                    st = d.Surf(V, F)
                    st_img = d.Surf(V, F, cn=d.corner_normals(V, F))
                    dv = np.abs(ctx.raw.signed(V)) / ctx.a
                    row["mesh_info"] = dict(verts=len(V), faces=len(F), to_raw_mean_a=float(dv.mean()),
                                            to_raw_max_a=float(dv.max()))
                    masks, seen = d.silhouettes(st, ctx.extent)
                    read_surface(ctx, row, "mesh", st, st_img, masks, seen, rng)
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)
            row["seconds"] = time.time() - t0
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            kind = "zb" if route == "pts" else "mesh"
            ikind = "zbs" if route == "pts" else "mesh"
            try:
                print(json.dumps(dict(mesh=mesh, route=route, label=label, seconds=round(row["seconds"], 1),
                                      biou_cor=row["sil"][kind]["cor"]["1024"]["held"]["biou2pct_err"],
                                      biou_raw=row["sil"][kind]["raw"]["1024"]["held"]["biou2pct_err"],
                                      f05_cor=row["d3"][kind]["cor"]["f0.5"], f05_raw=row["d3"][kind]["raw"]["f0.5"],
                                      psnr_cor=float(np.mean(row["img"][ikind]["cor"]["shaded"]["psnr"])),
                                      psnr_raw=float(np.mean(row["img"][ikind]["raw"]["shaded"]["psnr"])))), flush=True)
            except Exception:  # noqa: BLE001
                print(json.dumps(dict(mesh=mesh, route=route, label=label, error=row.get("error"))), flush=True)


if __name__ == "__main__":
    main()
