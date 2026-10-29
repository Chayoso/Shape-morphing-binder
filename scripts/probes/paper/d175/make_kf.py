"""make_kf.py OUT --tgt T --n N --seed S [--stretch 1.3] [--draws 8] -- D175: keyframe E of the pre-registration (tmp/d175/
PREREG.md): the run's own source sample (prepare's, bit for bit as pipeline_run.py builds it) and draws - 1 further draws of
the source mesh in the source's frame (seeds S + 1000 + 100 k, density-matched like the run's source), each stretched by A =
diag(1/sqrt(s), s, 1/sqrt(s)) about the run source's centroid (volume and centre kept, no rotation); an independent draw
(seed S + 2000), stretched the same way, is the 3D reference (`eval`), never read by the run. CPU only."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from scripts.pipeline_run import parse_args, prepared  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, load_normalized  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--stretch", type=float, default=1.3)
    ap.add_argument("--draws", type=int, default=8)
    a = ap.parse_args()
    args = parse_args(["--src", a.src, "--tgt", a.tgt, "--n", str(a.n), "--seed", str(a.seed)])
    _, prep = prepared(args, a.seed)
    src = np.asarray(prep.src, np.float32)
    k_src = prep.density["source_scale"] if prep.density else 1.0
    fsrc = {}
    s0 = load_normalized(a.src, a.n, a.seed, sample="stratified", frame=fsrc)
    assert np.array_equal((s0 * k_src).astype(np.float32), src), "the source frame does not reproduce the run's source"
    more = draws_in_frame(a.src, fsrc, a.n, [a.seed + 1000 + 100 * k for k in range(1, a.draws)] + [a.seed + 2000])
    draws = [src] + [(d * k_src).astype(np.float32) for d in more[:-1]]
    ev = (more[-1] * k_src).astype(np.float32)
    s = float(a.stretch)
    A = np.diag([1.0 / np.sqrt(s), s, 1.0 / np.sqrt(s)])
    c = src.astype(np.float64).mean(0)
    st = lambda x: ((x.astype(np.float64) - c) @ A.T + c).astype(np.float32)  # noqa: E731
    samples = np.stack([st(d) for d in draws])
    leash = 1.25 * float(max(np.abs(src).max(), np.abs(prep.tgt).max()))
    ext = 1.25 * float(np.abs(prep.render_x if prep.render_x is not None else prep.tgt).max())
    print(f"[kf] {a.out}: {len(samples)} samples of {a.n}, stretch {s} (det A {np.linalg.det(A):.6f}); centre "
          f"{np.round(c, 6).tolist()}, sample centres off by max {float(np.abs(samples.mean(1) - c).max()):.2e} wu; "
          f"max |x| keyframe {float(np.abs(samples).max()):.4f}, source {float(np.abs(src).max()):.4f}, target "
          f"{float(np.abs(prep.tgt).max()):.4f}; MPM leash {leash:.4f}, camera half-extent {ext:.4f}", flush=True)
    if float(np.abs(samples).max()) >= min(leash, ext):
        raise SystemExit("the keyframe leaves the camera box or the domain's leash")
    np.savez(a.out, samples=samples, eval=st(ev), A=A, centre=c, stretch=s, seed=a.seed, n=a.n,
             note=np.str_("D175 keyframe E: the run's source and draws stretched about its centroid (PREREG)"))


if __name__ == "__main__":
    main()
