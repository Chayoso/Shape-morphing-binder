"""d156h_floor.py OUT_DIR RUN_DIR MESH SEED [SEED ...] -- D156h2 (docs entry): the extraction-free reading of D156h. For
each run seed: the run's own render term (rebuilt as d156h_arrange.py rebuilds it: prepare's frame, the eight samples'
mean pictures at the run's resolution, the field's pitch, lattice and h) on the tag's 300k end frames of the full arm L
and its twin P, and on eight independent uniform samples of the mesh in the run's frame (seeds SEED + 2 + 100 k, the
evaluation's independent sample and seven more): the uniform floor, what a body of the mesh's own shape sampled
uniformly scores. One JSON line per seed in OUT_DIR/d156h2.jsonl. CPU."""
import json
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
from physmorph.losses.silhouette import set_kernel  # noqa: E402
from physmorph.pipeline.config import PipelineConfig  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, make_field  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, load_normalized  # noqa: E402

torch.set_grad_enabled(False)
cfg = PipelineConfig()
set_kernel("cic")
views = make_views(cfg.render_views, cfg.render_elevs)
AMB = cfg.pbr_ambient
N = 300000


def find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def one(out_dir, run_dir, mesh, seed):
    t0 = time.time()
    path = f"assets/{mesh}.obj"
    res = int(find_key(json.load(open(run_dir / f"{mesh}300k_L{seed}.json")), "render_res"))
    _, v_src = load_normalized("assets/isosphere.obj", N, seed, return_volume=True, sample="stratified", fill={})
    frame = {}
    tgt = load_normalized(path, N, seed + 1, match_volume=v_src, sample="stratified", frame=frame, fill={})
    zL = np.load(run_dir / f"{mesh}300k_L{seed}_end1.npz")
    zP = np.load(run_dir / f"{mesh}300k_P{seed}_end1.npz")
    assert np.array_equal(np.asarray(zL["tgt"], np.float32), tgt) and np.array_equal(np.asarray(zP["tgt"], np.float32), tgt)
    samples = [tgt] + draws_in_frame(path, frame, N, [seed + 1 + 100 * k for k in range(1, 8)])
    extent = float(np.abs(tgt).max()) * 1.25
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    T0 = torch.as_tensor(tgt)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(extent / res, 0.92 * pitch)

    def discs(x):
        p, g, _, _ = lattice.discs(make_field("zb", torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
        return p, nnf.normalize(g, dim=1)

    per = [exterior_targets(*discs(s), views, res, extent, cfg.sil_k, AMB) for s in samples]
    sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
    shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
    del per, samples

    def term(x):
        p, n = discs(x)
        s, b = d_exterior(p, n, sils, shade, views, res, extent, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        return dict(sil=float(s), pbr=float(b), total=float(s + cfg.w_pbr * b))

    aL = term(np.asarray(zL["frames"][-1], np.float32))
    aP = term(np.asarray(zP["frames"][-1], np.float32))
    fl = [term(x) for x in draws_in_frame(path, frame, N, [seed + 2 + 100 * k for k in range(8)])]
    tot = np.array([f["total"] for f in fl])
    floor = dict(total=float(tot.mean()), se=float(tot.std(ddof=1) / np.sqrt(len(tot))), min=float(tot.min()),
                 sil=float(np.mean([f["sil"] for f in fl])), pbr=float(np.mean([f["pbr"] for f in fl])),
                 all=[round(float(v), 7) for v in tot])
    lead = aP["total"] - aL["total"]
    row = dict(mesh=mesh, seed=seed, res=res, L=aL, P=aP, floor=floor, L_over_floor=aL["total"] / floor["total"],
               P_over_floor=aP["total"] / floor["total"], L_below_floor_z=(floor["total"] - aL["total"]) / floor["se"],
               L_below_min_draw=bool(aL["total"] < floor["min"]), lead=lead,
               lead_share_below_floor=(floor["total"] - aL["total"]) / lead if lead else None,
               seconds=round(time.time() - t0, 1))
    print(json.dumps(row), flush=True)
    with open(out_dir / "d156h2.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    od = Path(sys.argv[1])
    od.mkdir(parents=True, exist_ok=True)
    for sd in sys.argv[4:]:
        one(od, Path(sys.argv[2]), sys.argv[3], int(sd))
