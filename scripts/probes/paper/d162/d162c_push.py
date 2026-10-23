"""d162c_push.py OUT_DIR RUN [RUN ...] -- D162c (B), the GPU part: the render's push -dL_r/dx at kept states (16 a 40k
run, 12 a 300k run, evenly), L_r the render term (silhouette + w_pbr shading; with the particles' paint on the colour
scenes) on the run's own pictures, read as the window reads it (a field found at the state, the tracked discs,
d_exterior). The run's TargetPack is rebuilt from its provenance (pipeline_run.prepared / run_config, build_target) and
its target sample checked bit for bit. Saves OUT_DIR/push_RUN.npz (raws, the push per state, float32). Then
d162c_pushread.py decomposes on the CPU."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physmorph  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pipeline_run as PR  # noqa: E402
from d162c_motion import RUNS  # noqa: E402
from physmorph import gpu  # noqa: E402
from physmorph.pipeline.appearance import make_appearance  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior  # noqa: E402
from physmorph.pipeline.target import build_target  # noqa: E402
from physmorph.render.exterior import make_field, tracked  # noqa: E402


def run_args(name, prov):
    args = PR.parse_args([])
    for k, v in prov.items():
        if hasattr(args, k) and not isinstance(v, dict):
            setattr(args, k, v)
    args.appearance = "band" if "_band_" in name else ""
    return args


def push(tgt, cfg, x_np, albedo):
    e = tgt.ext
    x = gpu.tensor(np.asarray(x_np, np.float32)).requires_grad_(True)
    with torch.no_grad():
        discs = tracked(make_field(e.field, x.detach(), e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                        e.lattice, e.h, e.skin)
    if albedo is None:
        p, n, _ = discs.read(x)
        sil, pbr = d_exterior(p, n, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                              cfg.w_spray, cfg.pbr_ambient)
    else:
        p, n, _, c = discs.read(x, albedo, e.albedo_ref if e.albedo_ref is not None else albedo.mean(0))
        sil, pbr = d_exterior(p, n, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                              cfg.w_spray, cfg.pbr_ambient, albedo=c)
    L = sil + cfg.w_pbr * pbr
    g, = torch.autograd.grad(L, x)
    return (-g).detach().cpu().numpy().astype(np.float32), float(L)


def main(out_dir, names):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        _, prefix = RUNS[name]
        js = json.load(open(f"{prefix}.json"))
        args = run_args(name, js["provenance"])
        cfg0, prep = PR.prepared(args, args.seed)
        cfg = PR.run_config(args, cfg0, prep)
        z = np.load(f"{prefix}_frames12.npz", allow_pickle=True)
        assert np.array_equal(np.asarray(z["tgt"], np.float32), np.asarray(prep.tgt, np.float32)), name
        app = make_appearance(args.appearance, prep.src, prep.tgt, prep.tgt_draws) if args.appearance else None
        tgt = build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws, mesh=prep.tgt_mesh, appearance=app,
                           render_x=prep.render_x)
        alb = None if app is None else gpu.tensor(np.asarray(z["src_albedo"] if "src_albedo" in z else app.src, np.float32))
        raws = np.asarray(z["raws"])
        k = 12 if len(z["tgt"]) > 100000 else 16
        pick = sorted(set(np.linspace(0, len(raws) - 1, k).astype(int)))
        pushes, Ls = [], []
        for i in pick:
            v, L = push(tgt, cfg, z["frames"][i], alb)
            pushes.append(v)
            Ls.append(L)
        np.savez(out_dir / f"push_{name}.npz", raws=raws[pick], kept=np.asarray(pick), push=np.stack(pushes), L=np.asarray(Ls))
        print(f"[d162c B] {name}: {len(pick)} states, L_r {Ls[0]:.3e} -> {Ls[-1]:.3e}, {time.time() - t0:.0f} s", flush=True)
        del tgt, prep
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:] or list(RUNS))
