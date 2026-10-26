"""d156h_arrange.py OUT_DIR RUN_DIR MESH SEED -- D156h (docs entry): does the in-loop full arm lower its render
term by arrangement rather than by shape? On the tag's 300k end frames (RUN_DIR/{MESH}300k_{L,P}{SEED}_end1.npz) and an
independent mesh sample (seed SEED + 2), the run's own render term on the exterior, rebuilt as the run built it
(prepare's frame, the eight samples' mean pictures at the run's resolution, the field's pitch, lattice and h), read on
(a) the state's own particles and (b) a uniform resample by the run's sampler (eight seeds 95 + 100 k, the target
sample's number density) of the state's solid, extracted by D156f's delta = 0 rule (the exterior's zero-set mesh taken
back by the target sample's measured exterior offset c, marching cubes at half a pitch). One JSON line per state in
OUT_DIR/d156h.jsonl, then the mesh's summary line (the ratios, the lead, the shape share and its standard error). CPU."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physmorph  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as nnf  # noqa: E402
import d157_lib as D7  # noqa: E402
import open3d as o3d  # noqa: E402
import trimesh  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402
from physmorph.losses.silhouette import set_kernel  # noqa: E402
from physmorph.pipeline.config import PipelineConfig  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, make_field  # noqa: E402
from physmorph.sampling.mesh import (_oriented, draws_in_frame, fill_volume_in_frame, load_normalized,  # noqa: E402
                                     sample_in_frame)
from physmorph.sampling.orientation import orient_name  # noqa: E402

torch.set_grad_enabled(False)
NT = D7.NT
cfg = PipelineConfig()
set_kernel("cic")
views = make_views(cfg.render_views, cfg.render_elevs)
AMB = cfg.pbr_ambient
N = 300000
FINAL = [95 + 100 * k for k in range(8)]


def find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def main(out_dir, run_dir, mesh, seed):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = f"assets/{mesh}.obj"
    rj = json.load(open(run_dir / f"{mesh}300k_L{seed}.json"))
    res = int(find_key(rj, "render_res"))
    _, v_src = load_normalized("assets/isosphere.obj", N, seed, return_volume=True, sample="stratified", fill={})
    frame, fill_t = {}, {}
    tgt = load_normalized(path, N, seed + 1, match_volume=v_src, sample="stratified", frame=frame, fill=fill_t)
    z = np.load(run_dir / f"{mesh}300k_L{seed}_end1.npz")
    same_tgt = bool(np.array_equal(np.asarray(z["tgt"], np.float32), tgt))
    print(f"[d156h] {mesh} seed {seed}: render {res} px; the run's target sample reproduced bit for bit: {same_tgt}",
          flush=True)
    assert same_tgt
    samples = [tgt] + draws_in_frame(path, frame, N, [seed + 1 + 100 * k for k in range(1, 8)])
    extent = float(np.abs(tgt).max()) * 1.25
    sp_t = D7.median_kth_spacing(tgt, 8, 20000)
    pitch = 0.708 * sp_t
    T0 = torch.as_tensor(tgt)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(extent / res, 0.92 * pitch)
    print(f"[d156h] pitch {pitch:.4f} wu, lattice {h / pitch:.2f} pitches, extent {extent:.3f}", flush=True)

    def discs(x):
        p, g, _, _ = lattice.discs(make_field("zb", torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
        return p, nnf.normalize(g, dim=1)

    per = [exterior_targets(*discs(s), views, res, extent, cfg.sil_k, AMB) for s in samples]
    sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
    shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
    del per
    print(f"[d156h] pictures: the mean over {len(samples)} samples ({time.time() - t0:.0f} s)", flush=True)

    def term(x):
        p, n = discs(x)
        s, b = d_exterior(p, n, sils, shade, views, res, extent, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        return dict(sil=float(s), pbr=float(b), total=float(s + cfg.w_pbr * b), discs=int(len(p)))

    m = _oriented(path)
    Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64))
    if tc.volume < 0:
        tc.invert()
    true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
    zbt, _ = D7.field_mesh("zb", tgt, pitch)
    c_off = float(true.signed(zbt.V).mean())
    del zbt
    print(f"[d156h] the target sample's exterior offset c = {c_off / pitch:.3f} pitches", flush=True)
    tbox = (tgt.min(0) - 6 * pitch, tgt.max(0) + 6 * pitch)

    def extract(x, name):
        ext, info = D7.field_mesh("zb", x, pitch)
        hh = 0.5 * pitch
        # the grid: the exterior's box, cut to the target's box + 6 pitches (a stray piece far off adds nothing a
        # resample of the body could hold, and a box of the whole domain at half a pitch is 10^8 nodes)
        lo = np.maximum(ext.V.min(0) - 3 * pitch, tbox[0])
        hi = np.minimum(ext.V.max(0) + 3 * pitch, tbox[1])
        outside_box = float(((x < tbox[0]) | (x > tbox[1])).any(1).mean())
        nc = np.ceil((hi - lo) / hh).astype(int) + 1
        G = np.stack(np.meshgrid(*[lo[i] + hh * np.arange(nc[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        F = (-ext.signed(G) - c_off).reshape(*nc)
        del G
        v, f, _, _ = marching_cubes(F, level=0.0, spacing=(hh, hh, hh), allow_degenerate=False)
        S = trimesh.Trimesh(v + lo, f, process=True)
        S.update_faces(S.nondegenerate_faces())
        S.merge_vertices()
        if not S.is_watertight:
            trimesh.repair.fill_holes(S)
        if S.volume < 0:
            S.invert()
        o = out_dir / f"{mesh}_{name}_solid.obj"
        trimesh.Trimesh(S.vertices / frame["scale"] + frame["offset"], S.faces, process=False).export(str(o))
        assert orient_name(str(o)) == "id", o
        n_s = int(round(N * fill_volume_in_frame(str(o), frame, N) / fill_t["volume"]))
        vals = [term(xs) for xs in draws_in_frame(str(o), frame, n_s, FINAL)]
        tot = np.array([v["total"] for v in vals])
        return dict(volume=float(S.volume), watertight=bool(S.is_watertight), n=n_s, outside_target_box=outside_box,
                    total=float(tot.mean()), se=float(tot.std(ddof=1) / np.sqrt(len(tot))),
                    sil=float(np.mean([v["sil"] for v in vals])), pbr=float(np.mean([v["pbr"] for v in vals])),
                    all=[round(float(v), 7) for v in tot], ext_faces=info["faces"], out=str(o))

    rows = {}
    states = [("ref", sample_in_frame(path, frame, N, seed + 2)[0]),
              (f"P{seed}", np.asarray(np.load(run_dir / f"{mesh}300k_P{seed}_end1.npz")["frames"][-1], np.float32)),
              (f"L{seed}", np.asarray(z["frames"][-1], np.float32))]
    for name, x in states:
        ts = time.time()
        a = term(x)
        b = extract(x, name)
        row = dict(mesh=mesh, seed=seed, state=name, n=len(x), res=res, pitch=pitch, c_offset_pitch=c_off / pitch,
                   a=a, b=b, r=b["total"] / a["total"], vol_target=float(tc.volume), seconds=round(time.time() - ts, 1))
        rows[name] = row
        print(json.dumps(row), flush=True)
        with open(out_dir / "d156h.jsonl", "a") as fo:
            fo.write(json.dumps(row) + "\n")
    L, P, R = rows[f"L{seed}"], rows[f"P{seed}"], rows["ref"]
    da = P["a"]["total"] - L["a"]["total"]
    db = P["b"]["total"] - L["b"]["total"]
    se_db = float(np.hypot(P["b"]["se"], L["b"]["se"]))
    # r's standard error from b's (a is one state, read once)
    se_r = lambda w: w["b"]["se"] / w["a"]["total"]  # noqa: E731
    se_rd = float(np.hypot(se_r(L), se_r(P)))
    share = db / da if da != 0 else None
    summ = dict(mesh=mesh, seed=seed, summary=True, r_ref=R["r"], r_L=L["r"], r_P=P["r"], r_L_minus_r_P=L["r"] - P["r"],
                se_r_diff=se_rd, lead_a=da, lead_b=db, se_lead_b=se_db, shape_share=share,
                se_shape_share=(se_db / abs(da) if da else None), seconds=round(time.time() - t0, 1))
    if share is None:
        summ["reading"] = "no lead in (a)"
    elif L["r"] - P["r"] > 2 * se_rd and share < 0.5:
        summ["reading"] = "the full arm partly games the render through arrangement"
    elif abs(L["r"] - P["r"]) <= 2 * se_rd and share >= 0.8:
        summ["reading"] = "the lead is shape"
    else:
        summ["reading"] = "partly"
    print(json.dumps(summ), flush=True)
    with open(out_dir / "d156h.jsonl", "a") as fo:
        fo.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
