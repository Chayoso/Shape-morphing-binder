"""d162_arrange.py OUT_JSONL RUNS_FILE RUN_DIR NAME [NAME ...] -- D162, the coordinator's request of 2026-10-22 (from
D156h / D156h2: at the tag's 300k ends the full arm's render term lies below the uniform floor of the true mesh, so part of
the full-vs-twin gap is arrangement): per kept frame of a D162 40k run, how much of the render term's descent is
arrangement and how much is shape, read on the scale the floor-referenced weight reads (target.render_floor: the tracked
exterior of a field found at that state, d_exterior against the run's own pictures; on the colour scenes with the
particles' own paint).

For each run NAME (its flags from RUNS_FILE, its record RUN_DIR/NAME.json and RUN_DIR/NAME_frames12.npz) the run's
TargetPack is rebuilt as the run built it (pipeline_run.prepared / run_config, target.build_target; the target sample is
checked bit for bit against the archive's). Then, per kept frame (every second kept frame and the last):
  (a) the render term of the state's own particles (what the weight reads; its floor-referenced excess 1 - L_floor / a);
  (b) the mean render term of eight uniform resamples (the run's stratified sampler, seeds 95 + 100 k, at the target
      sample's number density) of the state's solid, extracted by D156h's rule: the exterior field's zero-set mesh at
      0.708 spacings (D157's field_mesh "zb") taken back by the floor sample's measured offset c to its true mesh,
      marching cubes at half a pitch; a resample point's paint is its nearest particle's.
(b) holds the state's shape and drops its arrangement; a - b is the arrangement's part. Reference rows: the floor sample
(the pictures' own first sample: (a) = L_floor, (b) its own resample = the resample's floor) and eight uniform samples of
the render target's true mesh at seeds seed + 2 + 100 k (D156h2's uniform floor). The run's per-window weight records
(render_L, render_floor, render_excess, lambda) of the window whose span holds the frame are attached. One JSON line per
row in OUT_JSONL. GPU (the render term), CPU (the extraction)."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

B = Path("/data/relcfd/chayo/physmorph_v2")
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "scripts"))
sys.path.insert(0, str(B / "tmp" / "d156"))
import physmorph  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402
import d157_lib as D7  # noqa: E402
import pipeline_run as PR  # noqa: E402
from physmorph.pipeline.appearance import make_appearance  # noqa: E402
from physmorph.pipeline.target import build_target, render_floor  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, fill_volume_in_frame, load_normalized, mesh_in_frame  # noqa: E402
from physmorph.sampling.orientation import orient_name  # noqa: E402

torch.set_grad_enabled(False)
FINAL = [95 + 100 * k for k in range(8)]


def windows_of(js):
    arm = list(js["arms"].values())[0]
    com = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")]
    return com


def at_frame(com, r):
    """The committed window whose span holds delivered frame r, and the frame its records were read at (its start)."""
    start = 0
    for h in com:
        if r < h["frame_end"]:
            return h, start
        start = h["frame_end"]
    return (com[-1], start) if com else (None, None)


def one(out, runs, run_dir, name):
    t0 = time.time()
    flags = runs[name]
    args = PR.parse_args(flags + ["--out", str(run_dir / name)])
    cfg0, prep = PR.prepared(args, args.seed)
    cfg = PR.run_config(args, cfg0, prep)
    z = np.load(run_dir / f"{name}_frames12.npz", allow_pickle=True)
    assert np.array_equal(np.asarray(z["tgt"], np.float32), np.asarray(prep.tgt, np.float32)), "target sample differs"
    app = make_appearance(args.appearance, prep.src, prep.tgt, prep.tgt_draws) if args.appearance else None
    tgt = build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws, mesh=prep.tgt_mesh, appearance=app,
                       render_x=prep.render_x)
    # the frame (prepare's, reproduced as it fills it) for the solids' resamples
    fill_src, fill_tgt = ({}, {}) if args.match_density else (None, None)
    _, v_src = load_normalized(args.src, args.n, args.seed, return_volume=True, sample="stratified", fill=fill_src)
    frame = {}
    tg = load_normalized(args.tgt, args.n, args.seed + 1, match_volume=v_src, sample="stratified", frame=frame,
                         fill=fill_tgt)
    assert np.array_equal(np.asarray(tg, np.float32), np.asarray(prep.tgt, np.float32)), "frame not reproduced"
    rt = args.render_tgt or args.tgt
    fx = prep.render_x if prep.render_x is not None else prep.tgt
    n_r = len(fx)
    alb_fx = None if app is None else (app.tgt if prep.render_x is None else app.albedo_at(np.asarray(fx)))
    src_alb = None if app is None else np.asarray(z["src_albedo"], np.float32)
    js = json.load(open(run_dir / f"{name}.json"))
    com = windows_of(js)
    L_floor_run = next((h.get("render_floor") for h in com if h.get("render_floor") is not None), None)
    if L_floor_run is None and (run_dir / f"{name}.log").exists():    # records before d162 0c: the run's log line
        for line in open(run_dir / f"{name}.log"):
            if "L_floor = " in line:
                L_floor_run = float(line.split("L_floor = ")[1].split()[0])
                break

    V, F = mesh_in_frame(rt, frame)
    tc, _ = D7.close_holes(V, F)
    if tc.volume < 0:
        tc.invert()
    true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
    pitch = 0.708 * D7.median_kth_spacing(np.asarray(fx, np.float64), 8, min(20000, n_r))
    zbt, _ = D7.field_mesh("zb", np.asarray(fx, np.float64), pitch)
    c_off = float(true.signed(zbt.V).mean())
    del zbt
    f_tgt = fill_volume_in_frame(args.tgt, frame, args.n)
    tbox = (np.asarray(fx).min(0) - 6 * pitch, np.asarray(fx).max(0) + 6 * pitch)
    work = out.parent / f"arrange_{name}"
    work.mkdir(parents=True, exist_ok=True)
    cache0 = os.environ.get("PHYSMORPH_CACHE")
    print(f"[d162a] {name}: pitch {pitch:.4f} wu, offset c {c_off / pitch:.3f} pitches, render target {rt}, "
          f"{n_r} floor particles; run's L_floor {L_floor_run}", flush=True)

    def term(x, alb):
        return render_floor(tgt, cfg, np.asarray(x, np.float32), alb)

    def resampled(x, tag, alb):
        ext, _ = D7.field_mesh("zb", np.asarray(x, np.float64), pitch)
        hh = 0.5 * pitch
        lo = np.maximum(ext.V.min(0) - 3 * pitch, tbox[0])
        hi = np.minimum(ext.V.max(0) + 3 * pitch, tbox[1])
        nc = np.ceil((hi - lo) / hh).astype(int) + 1
        G = np.stack(np.meshgrid(*[lo[i] + hh * np.arange(nc[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        Fd = (-ext.signed(G) - c_off).reshape(*nc)
        del G
        v, f, _, _ = marching_cubes(Fd, level=0.0, spacing=(hh, hh, hh), allow_degenerate=False)
        S = trimesh.Trimesh(v + lo, f, process=True)
        S.update_faces(S.nondegenerate_faces())
        S.merge_vertices()
        if not S.is_watertight:
            trimesh.repair.fill_holes(S)
        if S.volume < 0:
            S.invert()
        o = work / f"{tag}_solid.obj"
        trimesh.Trimesh(S.vertices / frame["scale"] + frame["offset"], S.faces, process=False).export(str(o))
        assert orient_name(str(o)) == "id", o
        os.environ["PHYSMORPH_CACHE"] = str(work / "cache")       # the solids' samples: scratch, deleted after
        try:
            n_s = int(round(args.n * fill_volume_in_frame(str(o), frame, args.n) / f_tgt))
            xs_all = draws_in_frame(str(o), frame, n_s, FINAL)
        finally:
            if cache0 is None:
                os.environ.pop("PHYSMORPH_CACHE", None)
            else:
                os.environ["PHYSMORPH_CACHE"] = cache0
        tree = cKDTree(np.asarray(x, np.float64)) if alb is not None else None
        vals = np.array([term(xs, None if alb is None else alb[tree.query(xs)[1]]) for xs in xs_all])
        return dict(b=float(vals.mean()), b_se=float(vals.std(ddof=1) / np.sqrt(len(vals))), n_s=n_s,
                    watertight=bool(S.is_watertight), b_all=[round(float(t), 7) for t in vals])

    def emit(row):
        print(json.dumps(row), flush=True)
        with open(out, "a") as fo:
            fo.write(json.dumps(row) + "\n")

    a_fx = term(fx, alb_fx)
    emit(dict(run=name, state="floor_sample", a=a_fx, L_floor_run=L_floor_run, **resampled(fx, "floor", alb_fx)))
    uni = [term(x, None if app is None else app.albedo_at(x))
           for x in draws_in_frame(rt, frame, n_r, [args.seed + 2 + 100 * k for k in range(8)])]
    emit(dict(run=name, state="uniform_true_mesh", a=float(np.mean(uni)), a_se=float(np.std(uni, ddof=1) / np.sqrt(8)),
              a_all=[round(float(t), 7) for t in uni]))
    raws = [int(r) for r in np.asarray(z["raws"])]
    pick = sorted(set(list(range(0, len(raws), 2)) + [len(raws) - 1]))
    for k in pick:
        x = np.asarray(z["frames"][k], np.float32)
        h, r0 = at_frame(com, raws[k])
        ts = time.time()
        a = term(x, src_alb)
        rs = resampled(x, f"f{raws[k]:05d}", src_alb)
        rec = {} if h is None else {key: h.get(key) for key in ("animation", "frame_end", "lambda", "lam_rule", "lam0",
                                                                  "render_L", "render_floor", "render_excess",
                                                                  "g_share", "transport_energy")}
        emit(dict(run=name, state="frame", raw=raws[k], t=raws[k] / 240.0, a=a,
                  excess_a=(max(0.0, 1.0 - L_floor_run / a) if L_floor_run else None), window=rec, window_start=r0,
                  seconds=round(time.time() - ts, 1), **rs))
    shutil.rmtree(work, ignore_errors=True)
    print(f"[d162a] {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    runs = {}
    for line in open(sys.argv[2]):
        p = line.split()
        if p and not p[0].startswith("#"):
            runs[p[0]] = p[1:]
    for nm in sys.argv[4:]:
        one(out, runs, Path(sys.argv[3]), nm)
