"""d172_detail_fast.py OUT_JSON SCENE_JSON [--all_frames] [--no_cache] [--refresh] -- D172's detail reader
(tmp/d172/d172_detail.py: the scale relief of a detailed / stylized mesh against the smoothed canvas) cut to the numbers the
summaries read, its readings unchanged (D173, 2026-10-29; the pattern of tmp/readers/README.md).

The old module is imported (it parses OUT_JSON SCENE_JSON from sys.argv as it always did) and its own functions run on the
same inputs: Ctx, layer, relief_read. What changes:
  - the kept frames: only those tmp/readers/fastlib.verdict_frames picks (the stop's two bracketing kept frames with a
    reading, the last kept frame and the last with a reading), the old reader read every third plus the stop's bracket
    and the last; d170_read.detail reads exactly those (sample_at at the stop, [-1]); --all_frames: the old set;
  - frames through a memory map of the archive (fastlib.Frames: the archive's own float32 bytes);
  - the context (frame, x_t, the canvas S, H, region, size row) and the reference readings cached per scene and seed
    (fastlib.Cache: arrays bit for bit, the readings as JSON).
The JSON has the old one's layout plus "fast" (frames read, stops, cache hits, stage seconds)."""
import json
from types import SimpleNamespace
import sys
import time
from pathlib import Path

FLAGS = [a for a in sys.argv[1:] if a.startswith("--")]
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith("--")]
ALL_FRAMES = "--all_frames" in FLAGS
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d172")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/readers")
import d172_detail as R  # noqa: E402  (parses OUT_JSON SCENE_JSON from sys.argv)
from fastlib import Cache, Clock, Frames, code_files, make_key, stop_raw, verdict_frames  # noqa: E402

np, D7, cKDTree = R.np, R.D7, R.cKDTree
scene, N, out_path = R.scene, R.N, R.out_path
CACHE = Cache(enabled="--no_cache" not in FLAGS, refresh="--refresh" in FLAGS)


def restore(arrays, objects, meta):
    c = R.Ctx.__new__(R.Ctx)
    c.frame, c.x_t, c.a = objects["frame"], arrays["x_t"], meta["a"]
    S = R.Surface.__new__(R.Surface)
    S.P, S.N = arrays["S_P"], arrays["S_N"]
    S.tree, S.n = cKDTree(S.P), len(S.P)
    c.S, c.H, c.region, c.s_bg, c.size, c.secs = S, arrays["H"], arrays["region"], meta["s_bg"], meta["size"], 0.0
    if "P" in arrays:
        c.P = arrays["P"]
    return c


def ctx_refs(seed, clock):
    import physmorph.sampling.mesh as SM
    from physmorph.sampling.mesh import draws_in_frame, fill_volume_in_frame, sample_in_frame
    files = [scene["tgt"], scene["smooth"], scene["detail"], "assets/isosphere.obj"]
    pp = scene["detail"] + ".pattern.npz"
    if Path(pp).exists():
        files.append(pp)
    key, parts = make_key("d172detail", files, code_files(SimpleNamespace(__file__=__file__), R, SM, sys.modules["fastlib"]),
                          N=N, seed=seed, s_bg=R.SBG, pattern_region=scene.get("pattern_region", True))
    hit = CACHE.load(key)
    if hit is not None:
        arrays, objects, meta = hit
        clock.mark(f"ctx+refs {seed} (cache)")
        return restore(arrays, objects, meta), meta["refs"], True
    c = R.Ctx(seed)
    clock.mark(f"ctx {seed}")
    # the references: d172_detail.main's lines, verbatim
    f_t = fill_volume_in_frame(scene["tgt"], c.frame, N)
    n_d = int(round(N * fill_volume_in_frame(scene["detail"], c.frame, N) / f_t))
    n_s = int(round(N * fill_volume_in_frame(scene["smooth"], c.frame, N) / f_t))
    refs = {"tgt": c.x_t, "detail_s1": sample_in_frame(scene["detail"], c.frame, n_d, seed + 1)[0],
            "detail_s2": draws_in_frame(scene["detail"], c.frame, n_d, [seed + 2])[0],
            "smooth_s2": draws_in_frame(scene["smooth"], c.frame, n_s, [seed + 2])[0]}
    rows = {}
    for k, X in refs.items():
        X = np.asarray(X, np.float64)
        rows[f"{k}{seed}"] = dict(relief=R.relief_read(c, X[R.layer(X)]))
    clock.mark(f"refs {seed}")
    arrays = dict(x_t=c.x_t, S_P=c.S.P, S_N=c.S.N, H=c.H, region=c.region)
    if hasattr(c, "P"):
        arrays["P"] = c.P
    CACHE.save(key, arrays, dict(frame=c.frame), dict(a=c.a, s_bg=c.s_bg, size=c.size, refs=rows), parts)
    return c, rows, False


def main():
    clock = Clock()
    res = dict(scene=scene, ctx={}, refs={}, runs={})
    seeds = sorted({int(lab.rsplit("_", 1)[-1]) for lab in scene["runs"]}) or list(scene.get("seeds", [97]))
    fast = dict(reader="d172_detail_fast", all_frames=ALL_FRAMES, cache_hit={}, frames_read={}, stops={}, secs={})
    ctxs = {}
    for seed in seeds:
        c, rows, hit = ctx_refs(seed, clock)
        ctxs[seed] = c
        res["ctx"][seed] = dict(c.size, s_bg_pitch=R.SBG, secs=c.secs)
        res["refs"].update(rows)
        fast["cache_hit"][seed] = hit
    stops = scene.get("stops", {})
    for label, prefix in scene["runs"].items():
        seed = int(label.rsplit("_", 1)[-1])
        c = ctxs[seed]
        fr = Frames(prefix + "_frames12.npz")
        assert np.abs(np.asarray(fr.get("tgt"), np.float64) - c.x_t).max() < 1e-5, f"{label}: the run's target is not the frame's"
        raws = fr.raws
        stop = stops.get(label, stop_raw(prefix))

        def read(j):
            X = np.asarray(fr[j], np.float64)
            return dict(raw=int(raws[j]), t=float(raws[j]) / 240.0, relief=R.relief_read(c, X[R.layer(X)]))
        if ALL_FRAMES:                                          # the old reader's frames: every third, the stop's, the last
            n = len(raws)
            js = set(range(0, n, 3)) | {n - 1}
            if stop is not None:
                k = int(np.searchsorted(raws, stop))
                js |= {max(0, k - 1), min(n - 1, k)}
            series = [read(j) for j in sorted(js)]
            idx = sorted(js)
        else:
            idx, rows = verdict_frames(raws, stop, read, lambda q: q["relief"].get("A") is not None)
            series = [rows[j] for j in idx]
        res["runs"][label] = dict(series=series)
        fast["frames_read"][label] = [int(raws[j]) for j in idx]
        fast["stops"][label] = stop
        clock.mark(f"run {label} ({len(idx)} frames)")
        print("run", label, json.dumps(series[-1]), flush=True)
    fast["secs"] = clock.secs
    res["fast"] = fast
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
