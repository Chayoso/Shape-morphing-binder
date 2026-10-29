"""d175_read.py OUT_DIR RUNDIR ASSETS [--title T] -- D175 reader of the pre-registered first experiment (tmp/d175/PREREG.md).
Runs on a card (the exterior operators are CUDA). Writes OUT_DIR/summary.md, verdict.json, rows.json and silhouettes.png,
and the top-level output/gpu/D175_SUMMARY.md.

Per run (RUNDIR/{D,S,T,N}_<seed>): the kept frames (frames12) and the JSON. Per seed the run's own final pack is rebuilt
(prepare + run_config + build_target, as pipeline_run.py builds it), and the keyframe's pictures are drawn from its
samples by target.keyframe_pictures (the run's operators) in the 18 run views and 12 held-out views (6 azimuths offset half
a step, elevations +-0.25). A body is read as the run reads it: a field found at the frame's state, its exterior discs, the
soft silhouette; alpha > 0.5 on both sides, IoU per view, the mean over views."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from scripts.pipeline_run import parse_args, prepared, run_config  # noqa: E402
from physmorph import gpu  # noqa: E402
from physmorph.losses.silhouette import soft_silhouette_multi  # noqa: E402
from physmorph.pipeline.target import build_target, keyframe_pictures, sample_pictures  # noqa: E402
from physmorph.render.exterior import make_field, tracked  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, load_normalized  # noqa: E402

HU = ["--lambda_ema", "0", "--settle_windows", "30", "--no-settle_relax", "--render_split", "u"]
NAMES = {"D": "dF_c + layer u + render loss (full), no keyframe", "S": "dF_c + layer u + render loss (full), storyboard",
         "T": "dF_c + layer u (twin), storyboard", "N": "naive picture swap (full)"}
RUNS = [("D", 97), ("D", 99), ("S", 97), ("S", 99), ("T", 97), ("N", 97)]
T_K = 240


def held_views():
    step = 2 * np.pi / 6
    return [(float(t + step / 2), e) for e in (0.25, -0.25) for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]


class Ctx:
    """A seed's final pack, its pictures and the keyframe's."""

    def __init__(self, seed, tgt_path, kf_path, src_path="assets/isosphere.obj", n=40000):
        args = parse_args(["--src", src_path, "--tgt", tgt_path, "--n", str(n), "--seed", str(seed), *HU])
        cfg0, prep = prepared(args, seed)
        self.cfg = cfg = run_config(args, cfg0, prep)
        self.final = f = build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws, render_x=prep.render_x)
        self.held = held_views()
        kf = np.load(kf_path)
        self.kf_eval = np.asarray(kf["eval"], np.float32)
        smp = list(np.asarray(kf["samples"], np.float32))
        self.P_run = torch.stack(keyframe_pictures(f, smp, cfg)[0]) > 0.5
        self.P_held = torch.stack(keyframe_pictures(f, smp, cfg, views=self.held)[0]) > 0.5
        self.F_run = torch.stack(list(f.ext.sils)) > 0.5               # the run's own target pictures
        # an independent sample of the target (seed + 2, prepare's evaluation seed), in the target's frame
        src0, v_src = load_normalized(args.src, args.n, seed, return_volume=True, sample="stratified", fill={})
        frame = {}
        t0, _ = load_normalized(tgt_path, args.n, seed + 1, match_volume=v_src, sample="stratified", return_volume=True,
                                frame=frame, fill={})
        assert np.array_equal(t0, prep.tgt), "the target frame does not reproduce the run's target"
        ind = gpu.tensor(draws_in_frame(tgt_path, frame, args.n, [seed + 2])[0])
        e = f.ext
        self.F_ind = torch.stack(sample_pictures(ind, cfg, e.pitch, e.radius, e.offset, e.lattice, e.h, f.views,
                                                 f.extent)[2][0]) > 0.5
        self.pitch = 0.708 * float(np.median(cKDTree(prep.tgt).query(prep.tgt, 9, workers=8)[0][:, 8]))

    def alpha(self, x, views):
        e, f = self.final.ext, self.final
        xt = gpu.tensor(np.asarray(x, np.float32))
        with torch.no_grad():
            d = tracked(make_field(e.field, xt, e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                        e.lattice, e.h, e.skin)
            p, _, _ = d.read(xt)
            return soft_silhouette_multi(p, views, self.cfg.render_res, f.extent, self.cfg.sil_k) > 0.5

    def occ(self, x):
        f = self.final
        ijk = torch.floor((gpu.tensor(np.asarray(x, np.float32)) - f.lgmin) / f.ldx).long()
        ok = ((ijk >= 0) & (ijk < torch.tensor(f.ldims, device=ijk.device))).all(1)
        g = torch.zeros(f.ldims, dtype=torch.bool, device=ijk.device)
        i = ijk[ok]
        g[i[:, 0], i[:, 1], i[:, 2]] = True
        return g


def iou(a, b):
    """Silhouettes (V, res, res): the mean over views of |a & b| / |a | b|."""
    return float(((a & b).flatten(1).sum(1).float() / (a | b).flatten(1).sum(1).clamp_min(1).float()).mean())


def iou3(a, b):
    """One occupancy grid: |a & b| / |a | b|."""
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


def pieces_run(fr, raws, pitch):
    """Serious pieces: kept frames spanning >= 120 frames (3 windows) in a row with a non-largest set (1.5-pitch link)
    of >= 10 particles. Returns (serious, the longest such span, the largest piece seen)."""
    has, biggest = [], 0
    for x in fr:
        x = np.asarray(x, np.float64)
        pairs = cKDTree(x).query_pairs(1.5 * pitch, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
        _, lab = connected_components(g, directed=False)
        sz = np.bincount(lab)
        sz[np.argmax(sz)] = 0
        m = int(sz.max()) if len(sz) else 0
        biggest = max(biggest, m)
        has.append(m >= 10)
    span, best, start = 0, 0, None
    for i, h in enumerate(has):
        if h and start is None:
            start = i
        if start is not None and (not h or i == len(has) - 1):
            end = i if h else i - 1
            best = max(best, int(raws[end] - raws[start]) + 12)
            start = None
    return best >= 120, best, biggest


def read_run(ctx, prefix):
    j = json.load(open(prefix + ".json"))
    a = next(iter(j["arms"].values()))
    z = np.load(prefix + "_frames12.npz")
    fr, raws = z["frames"], np.asarray(z["raws"])
    h, m, dn = a["history"], a["metrics"], int(a["deliver_n"])
    k = int(np.flatnonzero(raws == T_K)[0])
    x_k = fr[k]
    r = dict(kf_run=iou(ctx.alpha(x_k, ctx.final.views), ctx.P_run), kf_held=iou(ctx.alpha(x_k, ctx.held), ctx.P_held),
             kf_3d=iou3(ctx.occ(x_k), ctx.occ(ctx.kf_eval)))
    r["curve"] = [[int(raws[i]), round(iou(ctx.alpha(fr[i], ctx.final.views), ctx.P_run), 4)]
                  for i in range(len(raws)) if T_K - 80 <= raws[i] <= T_K + 80]
    end = fr[-1]
    r.update(fsil_ind=iou(ctx.alpha(end, ctx.final.views), ctx.F_ind), fsil_run=iou(ctx.alpha(end, ctx.final.views), ctx.F_run),
             chamfer=m["chamfer"], thin=m["thin_uncovered"], sil_iou_json=m["sil_iou"])
    com = [q for q in h if q.get("frame_end") and not q.get("null_commit")]
    last = next(q for q in com if q["frame_end"] == dn)
    opt = [q for q in com if not q.get("settle") and not q.get("hold")]
    r.update(v_end=last["v_mean"], jmin=min(q["Jmin_traj"] for q in com), windows=len(opt),
             holds=sum(int(bool(q.get("hold"))) for q in com), stop_fe=opt[-1]["frame_end"] if opt else None,
             minutes=a["seconds"] / 60.0, deliver_n=dn)
    lam = [q["lambda"] for q in opt if q.get("lambda") is not None]
    r["lam"] = float(np.median(lam)) if lam else None
    seg = [q for q in h if q.get("keyframe") == 0 and q.get("g_share") is not None]
    fin = [q for q in h if q.get("keyframe", None) is None and q.get("g_share") is not None and not q.get("settle")]
    r["g_share_seg"] = float(np.mean([q["g_share"] for q in seg])) if seg else None
    r["g_share_fin"] = float(np.mean([q["g_share"] for q in fin])) if fin else None
    gc = [q["g_raw_cos"] for q in h if q.get("keyframe") == 0 and q.get("g_raw_cos") is not None]
    r["g_raw_cos_seg"] = [round(g, 3) for g in gc]
    r["serious"], r["piece_span"], r["piece_max"] = pieces_run(fr, raws, ctx.pitch)
    r["_x_k"] = x_k
    return r


def fmt(v, f=".4f"):
    return "-" if v is None else format(v, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("rundir")
    ap.add_argument("assets")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--runs", default="", help="e.g. S97,T97 (default: the pre-registered six)")
    ap.add_argument("--kf", default="kf_s{seed}.npz")
    ap.add_argument("--prefix", default="", help="run name prefix in RUNDIR, e.g. show_")
    ap.add_argument("--t_k", type=int, default=240, help="the keyframe's frame")
    ap.add_argument("--title", default="D175: picture keyframe E (sphere stretched 1.3x in y at t = 1 s), sphere -> bunny 40k")
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    global T_K
    T_K = a.t_k
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [(r[0], int(r[1:])) for r in a.runs.split(",")] if a.runs else RUNS
    ctx = {s: Ctx(s, a.tgt, f"{a.assets}/{a.kf.format(seed=s)}", a.src, a.n) for s in sorted({s for _, s in runs})}
    R = {}
    for arm, s in runs:
        p = f"{a.rundir}/{a.prefix}{arm}_{s}"
        try:
            R[f"{arm}{s}"] = read_run(ctx[s], p)
        except (OSError, KeyError, StopIteration, IndexError, ValueError) as e:
            print(f"[read] {arm}_{s}: not read ({e!r})", flush=True)
    T = R.get("T97")
    D = [R[k] for k in ("D97", "D99") if k in R]
    V = {}
    for s in (97, 99):
        S, Ds = R.get(f"S{s}"), R.get(f"D{s}")
        if S is None or T is None or len(D) < 2 or Ds is None:
            V[f"S{s}"] = dict(pass_=False, why="INCOMPLETE")
            continue
        G_run = (S["kf_run"] - T["kf_run"]) / max(1 - T["kf_run"], 1e-9)
        G_held = (S["kf_held"] - T["kf_held"]) / max(1 - T["kf_held"], 1e-9)
        p1 = dict(G_run=G_run >= 0.6, G_held=G_held >= 0.5, iou3d=S["kf_3d"] >= T["kf_3d"] + 0.10,
                  vs_D=S["kf_run"] >= Ds["kf_run"] + 0.05)
        p2 = dict(fsil=S["fsil_ind"] >= min(d["fsil_ind"] for d in D) - 0.005,
                  chamfer=S["chamfer"] <= 1.1 * max(d["chamfer"] for d in D),
                  thin=S["thin"] <= max(d["thin"] for d in D) + 0.02)
        p3 = dict(v_end=S["v_end"] <= 1.5 * max(d["v_end"] for d in D), jmin=S["jmin"] > 0, pieces=not S["serious"])
        V[f"S{s}"] = dict(G_run=G_run, G_held=G_held, P1=p1, P2=p2, P3=p3,
                          pass_=all(p1.values()) and all(p2.values()) and all(p3.values()))
    N = R.get("N97")
    G_N = ((N["kf_run"] - T["kf_run"]) / max(1 - T["kf_run"], 1e-9)) if (N and T) else None
    ok = all(V[k]["pass_"] for k in ("S97", "S99"))
    L = [f"# {a.title}", "", "Pre-registration: tmp/d175/PREREG.md (2026-10-29 11:15 CDT). Method: storyboard ownership "
         "(design option C); lambda from the pre-pass at the source, held; inside the segment the held lambda on dF_c and u, "
         "after t_k Hu (held on u, the rule on dF_c).", "",
         "| run | KF-IoU run views | KF-IoU held-out | 3D IoU | final sil IoU (indep.) | chamfer | thin | v_end x1e3 | "
         "Jmin | serious pieces (span, max) | windows (holds) | stop frame | min | lambda | g_share seg / final |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in R.items():
        L.append(f"| {k} {NAMES[k[0]]} | {fmt(r['kf_run'])} | {fmt(r['kf_held'])} | {fmt(r['kf_3d'])} | "
                 f"{fmt(r['fsil_ind'])} | {fmt(r['chamfer'])} | {fmt(r['thin'], '.3f')} | {r['v_end'] * 1e3:.2f} | "
                 f"{r['jmin']:.3f} | {'YES' if r['serious'] else 'no'} ({r['piece_span']}, {r['piece_max']}) | "
                 f"{r['windows']} ({r['holds']}) | {r['stop_fe']} | {r['minutes']:.1f} | {fmt(r['lam'], '.4g')} | "
                 f"{fmt(r['g_share_seg'], '.3f')} / {fmt(r['g_share_fin'], '.3f')} |")
    L += ["", "## Verdict against the pre-registered bars", "",
          "| seed | G run (>= 0.6) | G held-out (>= 0.5) | 3D >= twin + 0.10 | KF >= D + 0.05 | final sil | chamfer | thin | "
          "v_end | Jmin > 0 | no serious piece | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "**no**"  # noqa: E731
    for k, v in V.items():
        if "P1" not in v:
            L.append(f"| {k} | {v['why']} |" + " - |" * 11)
            continue
        L.append(f"| {k} | {v['G_run']:.3f} {yn(v['P1']['G_run'])} | {v['G_held']:.3f} {yn(v['P1']['G_held'])} | "
                 f"{yn(v['P1']['iou3d'])} | {yn(v['P1']['vs_D'])} | {yn(v['P2']['fsil'])} | {yn(v['P2']['chamfer'])} | "
                 f"{yn(v['P2']['thin'])} | {yn(v['P3']['v_end'])} | {yn(v['P3']['jmin'])} | {yn(v['P3']['pieces'])} | "
                 f"{'PASS' if v['pass_'] else 'FAIL'} |")
    L += ["", "## Render influence", ""]
    if T is not None and "S97" in R:
        L.append(f"- lambda (held, the pre-pass at the source on the final target): {fmt(R['S97']['lam'], '.4g')}; "
                 f"g_share in the segment {fmt(R['S97']['g_share_seg'], '.3f')} (S97), after t_k "
                 f"{fmt(R['S97']['g_share_fin'], '.3f')}; D97 {fmt(R.get('D97', {}).get('g_share_fin'), '.3f')}.")
        L.append(f"- Twin vs full at frame 240: KF-IoU {T['kf_run']:.4f} (twin) vs {R['S97']['kf_run']:.4f} (S97)"
                 + (f" / {R['S99']['kf_run']:.4f} (S99)" if "S99" in R else "")
                 + f"; 3D IoU {T['kf_3d']:.4f} vs {R['S97']['kf_3d']:.4f}. The keyframe is reached only through the "
                 "render (the twin reads the same storyboard with lambda = 0 and stays the sphere).")
    if len(D) == 2:
        L.append(f"- The D seeds' spread (no keyframe): KF-IoU {abs(D[0]['kf_run'] - D[1]['kf_run']):.4f}, final sil IoU "
                 f"{abs(D[0]['fsil_ind'] - D[1]['fsil_ind']):.4f}, chamfer {abs(D[0]['chamfer'] - D[1]['chamfer']):.4f}.")
    if N is not None:
        neg = sum(int(g < 0) for g in N["g_raw_cos_seg"])
        L.append(f"- Naive arm N (pictures swapped, final geometry kept): G_N = {fmt(G_N, '.3f')} (prediction < 0.2); "
                 f"segment g_raw_cos {N['g_raw_cos_seg']} ({neg} < 0; prediction >= 4 of 6).")
    if "S97" in R:
        L.append(f"- S97 segment g_raw_cos (soundness vs render) {R['S97']['g_raw_cos_seg']}.")
    L += ["", "KF-IoU curve (run views) over frames 160-320: "
          + "; ".join(f"{k}: " + ", ".join(f"{f}:{v:.3f}" for f, v in r["curve"]) for k, r in R.items()), ""]
    try:
        figure(out / "silhouettes.png", ctx[97], R)
        L.append(f"Figure: {out}/silhouettes.png (frame 240, three run views; red = the keyframe P_k's outline).")
    except Exception as e:  # noqa: BLE001  (a figure must not cost the verdict)
        L.append(f"(figure not drawn: {e!r})")
    L += ["", f"VERDICT: {'PASS' if ok else 'FAIL'} (S97 {'PASS' if V['S97'].get('pass_') else 'FAIL'}, "
          f"S99 {'PASS' if V['S99'].get('pass_') else 'FAIL'})"]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    rows = {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in R.items()}
    json.dump(dict(rows=rows, G_N=G_N), open(out / "rows.json", "w"), indent=1, default=float)
    json.dump(dict(verdict="PASS" if ok else "FAIL", seeds=V), open(out / "verdict.json", "w"), indent=1, default=float)
    if a.summary:
        Path(a.summary).write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def figure(path, ctx, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vi = [0, 1, 6]
    keys = [k for k in ("T97", "S97", "N97", "D97") if k in R]
    fig, ax = plt.subplots(len(keys), len(vi), figsize=(3 * len(vi), 3 * len(keys)))
    ax = np.atleast_2d(ax)
    P = ctx.P_run.cpu().numpy()
    for i, k in enumerate(keys):
        A = ctx.alpha(R[k]["_x_k"], ctx.final.views).cpu().numpy()
        for j, v in enumerate(vi):
            ax[i, j].imshow(A[v].T, origin="lower", cmap="Greys", vmin=0, vmax=1.5)
            ax[i, j].contour(P[v].T.astype(float), levels=[0.5], colors="r", linewidths=1)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if j == 0:
                ax[i, j].set_ylabel(k)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
