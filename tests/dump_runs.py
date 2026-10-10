"""dump_runs.py DIR -- the runs of tests/test_dump_hooks.py, in one process under Warp's run-to-run deterministic
atomics and torch's deterministic algorithms (a CUDA run repeats bit for bit only so): the settled-contract cloud
without a hook twice, with --grad_dump and with --term_dump (into DIR). Prints one JSON line: whether each run is the
first one bit for bit (every delivered frame, the plastic deformation, every history field both record but the clocks), and the
dump files written."""
import json
import os
import sys

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
HOOK = ("active_set",)                             # --term_dump's own record (the active sets it dumps), null without it


def main(out_dir):
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)

    def run(**kw):
        cfg = PipelineConfig(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                             render_res=24, dt_res=32, patience=2, **kw)
        res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
        hist = [{k: json.dumps(v, default=str) for k, v in r.items() if k not in CLOCKS + HOOK} for r in res["history"]]
        return np.stack([np.asarray(x) for x in res["frames"].x]), np.asarray(res["Fp"]), hist

    def same(a, b):
        """Every delivered frame and Fp bit for bit, and every history field but the clocks and the hook's own record
        (--term_dump fills active_set)."""
        return (a[0].shape == b[0].shape and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                and len(a[2]) == len(b[2]) and all(ra[k] == rb[k] for ra, rb in zip(a[2], b[2]) for k in set(ra) & set(rb)))

    g, t = os.path.join(out_dir, "grad"), os.path.join(out_dir, "term")
    ref = run()
    rows = {"repeat": same(ref, run()), "grad": same(ref, run(grad_dump=g)), "term": same(ref, run(term_dump=t)),
            "committed": int(sum(1 for r in ref[2] if "frame_end" in r))}
    rows.update(grad_files=len(os.listdir(g)) if os.path.isdir(g) else 0,
                term_files=len(os.listdir(t)) if os.path.isdir(t) else 0)
    print(json.dumps(rows))


if __name__ == "__main__":
    main(sys.argv[1])
