"""appearance_runs.py -- D160's pipeline-level checks (tests/test_appearance_pipeline.py): the settled-contract cloud of
tests/refreeze_runs.py with the render terms on the exterior (render_exterior, the frozen recipe's), run under Warp's
run-to-run deterministic atomics and torch's deterministic algorithms (a CUDA run repeats bit for bit only so):
  plain   no appearance (the frozen path)
  white   appearance "white": one channel of ones against a white target (the reduction)
  twin0   render_weight_scale 0, no appearance
  twinA   render_weight_scale 0, paint A (two albedos, random by particle; the target red for x_0 > 0)
  twinB   render_weight_scale 0, paint B (another random paint; the target red for x_1 > 0)
  fullA   the render weight at its rule, paint A
  twinH   render_weight_scale 0, paint A, D160d's held colour weight (zero in the twin)
  fullH   paint A, D160d: the render terms the tag's, the colour term on the first window's weight held
  plain2  plain again (the deterministic mode holds)
Prints one JSON line: per run the trajectory digest (every delivered frame and the plastic deformation), the full digest
(the trajectory and every history field but the clocks) and the committed windows."""
import hashlib
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import warp as wp  # noqa: E402

wp.config.deterministic = wp.config.DeterministicMode.RUN_TO_RUN
wp.config.deterministic_max_records = 128
import physmorph  # noqa: F401,E402  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
from physmorph.mpm.state import MPMParams  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from physmorph.pipeline.appearance import RED, WHITE, Appearance, make_appearance  # noqa: E402

CLOCKS = ("t_attempt", "t_start", "t_grad", "t_ls", "t_commit", "prof_run", "seconds")
REFREEZE = dict(relax_driven=True, u_adam_carry=True, spacing_symmetric=True, min_spacing=0.9, volume_exact="carried",
                assim_volume=True, lambda_ema=1.0, render_exterior=True)


def cloud():
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


def paint(seed, src, tgt, axis):
    rng = np.random.default_rng(seed)
    two = lambda red: np.where(red[:, None], np.asarray(RED, np.float32), np.asarray(WHITE, np.float32))  # noqa: E731
    return Appearance(f"paint{seed}", two(rng.uniform(size=len(src)) < .5), two(tgt[:, axis] > 0), [], {})


def run(appearance=None, scale=1.0, weight="rule"):
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    src, tgt = cloud()
    cfg = PipelineConfig(T=4, iters=2, animations=4, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_weight_scale=scale, appearance_weight=weight,
                         **REFREEZE)
    app = appearance(src, tgt) if callable(appearance) else appearance
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None, appearance=app)
    hist = [json.dumps({k: v for k, v in r.items() if k not in CLOCKS}, sort_keys=True, default=str)
            for r in res["history"]]
    frames = np.stack([np.asarray(x) for x in res["frames"].x])
    h = hashlib.sha256()
    h.update(frames.tobytes())
    h.update(np.asarray(res["Fp"]).tobytes())
    traj = h.hexdigest()
    for s in hist:
        h.update(s.encode())
    return dict(traj=traj, full=h.hexdigest(),
                committed=sum(1 for r in res["history"] if r.get("frame_end") and not r.get("null_commit")))


if __name__ == "__main__":
    out = dict(plain=run(), white=run(lambda s, t: make_appearance("white", s, t)),
               twin0=run(None, 0.0), twinA=run(lambda s, t: paint(1, s, t, 0), 0.0),
               twinB=run(lambda s, t: paint(2, s, t, 1), 0.0), fullA=run(lambda s, t: paint(1, s, t, 0)),
               twinH=run(lambda s, t: paint(1, s, t, 0), 0.0, "held"),
               fullH=run(lambda s, t: paint(1, s, t, 0), 1.0, "held"),
               plain2=run())
    print(json.dumps(out))
