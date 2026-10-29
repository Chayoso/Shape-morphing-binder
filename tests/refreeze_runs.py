"""refreeze_runs.py -- the pipeline-level pin of the re-freeze (tests/test_refreeze_reference.py): the settled-contract
cloud of tests/dump_runs.py run twice in one process under Warp's run-to-run deterministic atomics and torch's
deterministic algorithms (a CUDA run repeats bit for bit only so), with the re-freeze's run-stage switches: the layer
relaxation in the driven steps (D146), u's Adam moments carried (D146b), the minimum spacing 0.9 on the symmetric
closure (D149), the volume carried and assimilated (D131, D135) and the render weight at its rule every window (D120).
Prints one JSON line: whether the two runs repeat bit for bit, the committed windows, and a SHA-256 digest of every
delivered frame, the plastic deformation and every history field but the clocks."""
import hashlib
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
from physmorph.mpm.state import MPMParams  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402

CLOCKS = ("t_attempt", "t_start", "t_grad", "t_ls", "t_commit", "prof_run", "seconds")
# records added after the pin (read only, nothing enters a step; wip/final, D178): D173's per-block render shares and
# D164's replay records on the windows the earlier check would have nulled
RECORDS = ("lam_rule_block", "g_share_dfc", "g_share_u", "g_rend_u_norm", "g_phys_u_norm", "g_rend_dfc_norm", "g_phys_dfc_norm",
           "lam_block_dfc", "lam_block_u", "u_render_nonzero", "replay_rescued", "replay_noise", "replay_res", "replay_pairs",
           "replay_taped_max", "commit_mismatch", "commit_tol", "commit_replay_start", "commit_reads_accepted",
           "lam_mask", "g_mask_dfc_norm", "g_share_dfc_masked")
REFREEZE = dict(relax_driven=True, u_adam_carry=True, spacing_symmetric=True, min_spacing=0.9, volume_exact="carried",
                assim_volume=True, lambda_ema=1.0)


def run():
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    cfg = PipelineConfig(T=4, iters=2, animations=4, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, **REFREEZE)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    hist = [json.dumps({k: v for k, v in r.items() if k not in CLOCKS + RECORDS}, sort_keys=True, default=str)
            for r in res["history"]]
    frames = np.stack([np.asarray(x) for x in res["frames"].x])
    h = hashlib.sha256()
    h.update(frames.tobytes())
    h.update(np.asarray(res["Fp"]).tobytes())
    for s in hist:
        h.update(s.encode())
    return h.hexdigest(), sum(1 for r in res["history"] if r.get("frame_end") and not r.get("null_commit"))


if __name__ == "__main__":
    a, committed = run()
    b, _ = run()
    print(json.dumps({"digest": a, "repeat": a == b, "committed": committed}))
