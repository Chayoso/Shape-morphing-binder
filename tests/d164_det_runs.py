"""d164_det_runs.py -- the run of tests/test_d164_replay.py's deterministic case: the settled-contract cloud with the
re-freeze's switches (tests/refreeze_runs.py), three iterations a window, in one process under Warp's run-to-run
deterministic atomics and torch's deterministic algorithms. Prints one JSON line: per window the start pair, the
largest taped pair, the pair count, the commit mismatch and the null reason (D164: with the atomics' order fixed, every
replay pair the window measures and the commit must agree bit for bit; only the atomics separate them)."""
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import warp as wp  # noqa: E402

wp.config.deterministic = wp.config.DeterministicMode.RUN_TO_RUN
wp.config.deterministic_max_records = 128          # the layer relaxation's neighbour loop is not bounded statically
import physmorph  # noqa: F401,E402  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
import physmorph.pipeline.run.runner as runner_mod  # noqa: E402
from physmorph.mpm.state import MPMParams  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402

REFREEZE = dict(relax_driven=True, u_adam_carry=True, spacing_symmetric=True, min_spacing=0.9, volume_exact="carried",
                assim_volume=True, lambda_ema=1.0)
KEYS = ("accepted", "replay_rel", "replay_taped_max", "replay_pairs", "commit_mismatch", "null_reason")


def main():
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    optimize, rows = runner_mod.optimize_window, []

    def window(*a, **k):
        res = optimize(*a, **k)
        rows.append({key: res.stats.get(key) for key in KEYS})
        return res

    runner_mod.optimize_window = window
    cfg = PipelineConfig(T=4, iters=3, animations=4, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, **REFREEZE)
    run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    print(json.dumps({"windows": rows}))


if __name__ == "__main__":
    main()
