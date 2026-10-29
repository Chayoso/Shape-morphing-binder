"""d154_ind_frame.py OUT_JSONL MESH:SEED [...] -- D154 aside, D152's open item (seed 99's yardstick floor 1.4-1.7x on all
six meshes, cause not found): is it the target frame? Each mesh-seed's target frame (mesh_frames.py) differs in scale
by up to 0.2 % (load_normalized normalises each seed's own sample). D90's independent sample IND is taken to be in seed
97's frame (its offset to the mesh matches seed 97's own sample, D154 (1)). Here the yardstick's floor (the run's own
target sample read by render_terms_probe's exterior operators against IND's pictures, 96 px; CPU port of repo_r77's
code) is read twice per mesh-seed: against IND as it is, and against IND mapped into the seed's frame,
x' = (x / scale_97 + offset_97 - offset_s) x scale_s. CPU; measurement only."""
import json
import sys

import numpy as np

B = "/data/relcfd/chayo/physmorph_v2"
G = f"{B}/output/gpu"
sys.path.insert(0, f"{B}/repo_r77")
import torch  # noqa: E402
import torch.nn.functional as nnf  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from physmorph.losses.silhouette import set_kernel  # noqa: E402
from physmorph.pipeline.config import PipelineConfig  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(8)
cfg = PipelineConfig()
res = cfg.render_res_hi
set_kernel("cic")
views = make_views(cfg.render_views, cfg.render_elevs)


def kth(x, k=8, sub=20000):
    s = x[np.random.default_rng(0).choice(len(x), sub, replace=False)]
    return float(np.median(cKDTree(s).query(s, k + 1)[0][:, k].astype(np.float32).astype(np.float64))) * (sub / len(x)) ** (1 / 3)


def floor(own, ind):
    tgt = torch.as_tensor(ind)
    extent = float(tgt.abs().max()) * 1.25
    pitch = .708 * kth(ind)
    center = tgt.mean(0)
    lat = Lattice(center, 2.8 * float((tgt - center).norm(dim=1).max()))
    h = min(extent / res, .92 * pitch)

    def discs(x):
        p, g, _, _ = lat.discs(ZhuBridson(x, pitch), h, refine=False)
        return p, nnf.normalize(g, dim=1)

    e = exterior_targets(*discs(tgt), views, res, extent, cfg.sil_k, cfg.pbr_ambient)
    s, p = d_exterior(*discs(torch.as_tensor(own)), *e, views, res, extent, cfg.sil_k, cfg.w_hole, cfg.w_spray, cfg.pbr_ambient)
    return float(s), float(p)


fr = {(r["mesh"], r["seed"]): r for r in map(json.loads, open(f"{G}/d154/mesh_frames.jsonl"))}
with open(sys.argv[1], "a") as out:
    for spec in sys.argv[2:]:
        mesh, seed = spec.split(":")
        seed = int(seed)
        ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
        own = np.asarray(np.load(f"{G}/d145t_b1/{mesh}300k_L{seed}_end1.npz")["tgt"], np.float32)
        a, b = fr[(mesh, 97)], fr[(mesh, seed)]
        mapped = ((ind / a["scale"] + np.asarray(a["offset"]) - np.asarray(b["offset"])) * b["scale"]).astype(np.float32)
        r = dict(mesh=mesh, seed=seed, scale_ratio=b["scale"] / a["scale"],
                 shift_wu=float(np.linalg.norm((np.asarray(a["offset"]) - np.asarray(b["offset"])) * b["scale"])),
                 as_is=floor(own, ind.astype(np.float32)), mapped=floor(own, mapped))
        out.write(json.dumps(r) + "\n")
        out.flush()
        print(json.dumps(r), flush=True)
