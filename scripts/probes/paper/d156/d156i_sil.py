"""d156i_sil.py OUT_DIR HULL_DIR N RES MESH -- D156i's silhouette diagnostic (after its reading): where does the visual
hull's silhouette term exceed the floor? D156g's frame and pictures (the mean over the eight samples of the mesh); the
alpha pictures (exterior_targets' silhouettes) of the hull file's eight held-out draws (seeds 95 + 100 k) and of the
mesh's eight floor draws (99 + 100 k), each averaged over its draws; per view and summed, the hole mass (target alpha
above the state's: material missing) and the spray mass (the state's above the target's: material in excess), in pixels.
One JSON line per mesh in OUT_DIR/d156i_sil.jsonl. CPU."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physmorph  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as nnf  # noqa: E402
import d156_photohull as PH  # noqa: E402
from d156_photohull import D7  # noqa: E402
from physmorph.pipeline.render_loss import exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, make_field  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, fill_volume_in_frame, load_normalized  # noqa: E402

torch.set_grad_enabled(False)
cfg, AMB, views, SEED = PH.cfg, PH.AMB, PH.views, PH.SEED


def main(out_dir, hull_dir, N, RES, mesh):
    path, hpath = f"assets/{mesh}.obj", str(hull_dir / f"{mesh}_hull18_r{RES}.obj")
    _, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
    frame = {}
    tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", frame=frame)
    box = float(np.abs(tgt).max()) * 1.25
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    T0 = torch.as_tensor(tgt)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(box / RES, 0.92 * pitch)

    def alpha(xs):
        per = []
        for x in xs:
            p, g, _, _ = lattice.discs(make_field("zb", torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
            per.append(exterior_targets(p, nnf.normalize(g, dim=1), views, RES, box, cfg.sil_k, AMB)[0])
        return [torch.stack(v).mean(0).numpy().astype(np.float64) for v in zip(*per)]

    target = alpha([tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)]))
    n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
    states = dict(hull_file=alpha(draws_in_frame(hpath, frame, n_vh, [95 + 100 * k for k in range(8)])),
                  floor=alpha(draws_in_frame(path, frame, N, [SEED + 2 + 100 * k for k in range(8)])))
    row = dict(mesh=mesh, res=RES, views=len(views))
    for name, A in states.items():
        hole = [float(np.clip(t - a, 0, None).sum()) for t, a in zip(target, A)]
        spray = [float(np.clip(a - t, 0, None).sum()) for t, a in zip(target, A)]
        row[name] = dict(hole_px=float(sum(hole)), spray_px=float(sum(spray)), hole_per_view=[round(v, 1) for v in hole],
                         spray_per_view=[round(v, 1) for v in spray], target_px=float(sum(float(t.sum()) for t in target)))
    print(json.dumps(row), flush=True)
    with open(out_dir / "d156i_sil.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
