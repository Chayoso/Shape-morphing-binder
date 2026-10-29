"""make_kf_letter.py OUT --src S --tgt T --n N --seed SEED [--letter S] [--depth_frac 0.3] [--draws 8] -- D175 showcase: a
letter keyframe. The letter (DejaVu Sans Bold, matplotlib's TextPath) is extruded along z and sampled as a jittered grid at
the run's source density: n points over the source's volume, so the volume is kept. Its depth is depth_frac x its height.
It is centred at the run source's centroid, so the centre of mass is kept. A letter is no rigid copy of the source, so the
rotation condition holds by construction.

The letter lies in the xy-plane facing +z: it reads in the run view at azimuth 0, elevation 0, which looks along -z with x
right and y up. Writes 'samples' (draws, n_k, 3), with seeds SEED + 1000 + 100 k for the jitter, and an independent 'eval'
(seed SEED + 2000). CPU sampling; prepare reads the card."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from scripts.pipeline_run import parse_args, prepared  # noqa: E402


def letter_mask(ch):
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MPath
    from matplotlib.textpath import TextPath
    tp = TextPath((0, 0), ch, size=1.0, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    v = tp.vertices
    lo, hi = v.min(0), v.max(0)
    path = MPath((v - lo) / (hi[1] - lo[1]), tp.codes)          # height 1, the width in proportion
    return path, (hi[0] - lo[0]) / (hi[1] - lo[1])


def sample(path, width, height, depth, h, centre, seed):
    """A jittered grid of pitch h inside the extruded letter (height `height`, depth `depth`), centred at `centre`."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = (int(np.ceil(s / h)) for s in (width * height, height, depth))
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    p = (g + rng.random(g.shape)) * h
    keep = path.contains_points(p[:, :2] / height) & (p[:, 2] < depth)
    p = p[keep]
    return (p - p.mean(0) + centre).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src", default="assets/bunny.obj")
    ap.add_argument("--tgt", default="assets/dragon.obj")
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--letter", default="S")
    ap.add_argument("--depth_frac", type=float, default=0.3)
    ap.add_argument("--draws", type=int, default=8)
    a = ap.parse_args()
    args = parse_args(["--src", a.src, "--tgt", a.tgt, "--n", str(a.n), "--seed", str(a.seed)])
    _, prep = prepared(args, a.seed)
    src = np.asarray(prep.src, np.float64)
    vol = float(prep.density["source_fill"] * prep.density["source_scale"] ** 3) if prep.density else float(prep.v_src)
    h = (vol / a.n) ** (1.0 / 3.0)
    path, wr = letter_mask(a.letter)
    g = np.stack(np.meshgrid(np.linspace(0, wr, 600), np.linspace(0, 1, 600), indexing="ij"), -1).reshape(-1, 2)
    area = float(path.contains_points(g).mean()) * wr            # the letter's area at height 1
    height = (vol / (area * a.depth_frac)) ** (1.0 / 3.0)
    depth = a.depth_frac * height
    c = src.mean(0)
    seeds = [a.seed + 1000 + 100 * k for k in range(a.draws)] + [a.seed + 2000]
    S = [sample(path, wr, height, depth, h, c, s) for s in seeds]
    samples, ev = S[:-1], S[-1]
    n_k = min(len(s) for s in samples)
    samples = np.stack([s[np.random.default_rng(i).permutation(len(s))[:n_k]] for i, s in enumerate(samples)])
    leash = 1.25 * float(max(np.abs(src).max(), np.abs(prep.tgt).max()))
    ext = 1.25 * float(np.abs(prep.tgt).max())
    print(f"[kf] {a.out}: letter {a.letter!r}, height {height:.4f}, width {wr * height:.4f}, depth {depth:.4f} wu, pitch "
          f"{h:.5f}; {len(samples)} samples of {n_k} (source {a.n}), volume {n_k * h ** 3:.4f} vs the source's {vol:.4f}; "
          f"centre off by {float(np.abs(samples.mean(1) - c).max()):.2e}; max |x| {float(np.abs(samples).max()):.4f}, "
          f"MPM leash {leash:.4f}, camera half-extent {ext:.4f}", flush=True)
    if float(np.abs(samples).max()) >= min(leash, ext):
        raise SystemExit("the keyframe leaves the camera box or the domain's leash")
    np.savez(a.out, samples=samples, eval=ev, centre=c, letter=np.str_(a.letter), height=height, depth=depth, seed=a.seed,
             n=a.n, note=np.str_("D175 showcase: an extruded letter at the source's volume and centre"))


if __name__ == "__main__":
    main()
