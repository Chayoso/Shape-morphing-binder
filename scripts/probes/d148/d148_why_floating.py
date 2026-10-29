"""d148_why_floating.py LABEL FRAMES12 RUN_JSON OUT_PREFIX VIEW:TRACKS_JSON:FRAMES_DIR:AZ [VIEW:...] -- why does the
objective let a visible floating piece survive? (CPU; the coordinator's list, 2026-10-15)

Pieces: the detector's tracks (d140_image.py on the run's renders: VIEW side az 90, front az 35) with particles apart
projecting into them (cls_i > 0), each confirmed at its largest frame by the display's own field (visible_apart.py's
per-set classifier: a blob, not occluded from that camera, projecting into the track's box); the piece is the union of
those blobs' connected sets; tracks sharing particles merge. Per piece: frames, site (box), particles, size, fate (the
share of its particles in the body's largest set at the run's last frame).

Per piece and window while it is out (the kept frame nearest the window's start), over its particles (median, and the
share where it matters):
 (a) the signed distance to the true target mesh (open3d's exact distance on the oriented, framed mesh of prepare's
     load_normalized; sign by a 3-ray vote), in pitches (+ outside);
 (b) the spray cleanup's field as target.py builds it (t.dt3: the CIC target grid at dt_res 160 over 3 extents, its EDT
     clamped at dt_clamp_frac extents) at the particles: value (pitches) and |grad|; the isolation gate (> 0 = counted);
 (c) the near band: the distance to the nearest target point (pitches), eligible (berth < d < loss cell), and whether
     that point is on the piece's own side: the segment from the target point nearest the piece's origin (its
     particles' positions 0.25 s before the piece's first frame) to it stays inside the mesh;
 (d) the transport: the debiased map's displacement on the gate grid (objective.py's u-gate solve, as d148_wl_forensic)
     at the piece against the body's median, and its cosine with the direction to the nearest target point;
 (e) the 18 training views at the run's render_res (orthographic, extent 1.25 x the target's half-size, soft
     silhouettes alpha = 1 - exp(-1.5 count), CIC): the views where the piece raises alpha by more than 0.01 at a pixel
     where the target's alpha is below 0.5 (the excess term can see it), and the largest rise anywhere.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.getcwd())
import physmorph  # noqa: F401  (before torch)
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "probes", "paper"))
import visible_apart as VA  # noqa: E402
from physmorph.losses.grid_ot import grid_transport_displacement  # noqa: E402
from physmorph.losses.volumetric import d_w1, target_mass_grid  # noqa: E402
from physmorph.pipeline.render_loss import make_views  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402
from physmorph.sampling.mesh import _oriented  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
label, fpath, jpath, out_prefix = sys.argv[1:5]
views_in = [v.split(":") for v in sys.argv[5:]]

z = np.load(fpath)
fr, raws = z["frames"], np.asarray(z["raws"])
tgt = np.asarray(z["tgt"], np.float64)
run = json.load(open(jpath))
prov = run["provenance"]
cfgj = next(iter(run["arms"].values()))["config"]
a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
mpm = prov["mpm"]
NX, DXM, GMIN = int(mpm["nx"]), float(mpm["dx"]), np.asarray(mpm["grid_min"], np.float64)

# (a) the true target mesh in the run's frame
src_x, v_src = load_normalized(prov["src"], int(prov["n"]), int(prov["seed"]), return_volume=True, sample="stratified",
                               fill={})
frame = {}
tgt2, _ = load_normalized(prov["tgt"], int(prov["n"]), int(prov["seed"]) + 1, match_volume=v_src, sample="stratified",
                          return_volume=True, frame=frame, fill={})
print(f"[{label}] the recomputed target sample against the run's: max |d| {np.abs(tgt2 - tgt).max():.2e} wu", flush=True)
mesh = _oriented(prov["tgt"])
import open3d as o3d  # noqa: E402
V = (np.asarray(mesh.vertices, np.float64) - frame["offset"]) * frame["scale"]
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))


def sdist(p):
    return scene.compute_signed_distance(o3d.core.Tensor(np.asarray(p, np.float32)), nsamples=3).numpy() / a


# (b) the cleanup's field (target.py)
extent = float(np.abs(tgt).max()) * 1.25
dt_res, clampf = int(cfgj.get("dt_res", 160)), float(cfgj.get("dt_clamp_frac", 2.0))
dtdx = 3.0 * extent / dt_res
dtgmin = torch.full((3,), -1.5 * extent, dtype=torch.float32)
tt = torch.as_tensor(tgt, dtype=torch.float32)
g3 = target_mass_grid(tt, torch.ones(len(tt)), dtgmin, dtdx, (dt_res,) * 3)
occ = (g3 > 1e-6).reshape((dt_res,) * 3).numpy()
dt3 = torch.as_tensor(np.minimum(ndimage.distance_transform_edt(~occ) * dtdx, clampf * extent).astype(np.float32)
                      ).reshape(-1)
print(f"[{label}] cleanup field: cell {dtdx:.4f} wu = {dtdx / a:.2f} pitches; occupied cells {int(occ.sum())}", flush=True)


def dt_at(p):
    x = torch.as_tensor(np.asarray(p, np.float32)).requires_grad_(True)
    vals = []
    for i in range(len(x)):
        vals.append(d_w1(x[i:i + 1], torch.ones(1), dt3, dtgmin, dtdx, (dt_res,) * 3))
    v = torch.stack(vals)
    g = torch.autograd.grad(v.sum(), x)[0]
    return v.detach().numpy() / a, g.norm(dim=1).numpy()


# (c) the near band
ttree = cKDTree(tgt)
nn_sp = float(np.median(ttree.query(tgt, 2)[0][:, 1]))
berth = float(cfgj.get("nn_berth_k", 1.0)) * nn_sp
dmin32 = GMIN.astype(np.float32)
span = float((dmin32 + DXM * np.full(3, NX, np.float32) - dmin32).max())
ldx = span / int(cfgj["loss_res"])
gdx = span / NX


def inside_segment(p0, p1, n=9):
    ts = np.linspace(0, 1, n)[:, None]
    return bool((sdist(p0[None] + ts * (p1 - p0)[None]) <= 0.0).all())


# (d) the transport's map on the gate grid
ggrid = target_mass_grid(tt, torch.ones(len(tt)), torch.as_tensor(dmin32), gdx, (NX,) * 3)


def transport(x):
    xt = torch.as_tensor(np.asarray(x, np.float32))
    return grid_transport_displacement(xt, torch.ones(len(xt)), ggrid, torch.as_tensor(dmin32), gdx, (NX,) * 3,
                                       eps=gdx ** 2, iters=1600, tol=0.01).numpy()


# (e) the training views
VIEWS = make_views(int(cfgj.get("render_views", 6)), tuple(cfgj.get("render_elevs", (0.0, 0.5, -0.5))))
RES = int(cfgj["render_res"])


def counts(x, th, phi):
    right = np.array([np.cos(th), 0.0, -np.sin(th)])
    up = np.array([-np.sin(phi) * np.sin(th), np.cos(phi), -np.sin(phi) * np.cos(th)])
    rel = (np.stack([x @ right, x @ up], 1) + extent) / (2 * extent) * RES
    base = np.floor(rel).astype(np.int64)
    fr_ = rel - base
    img = np.zeros(RES * RES)
    for ox in (0, 1):
        wx = fr_[:, 0] if ox else 1 - fr_[:, 0]
        for oy in (0, 1):
            wy = fr_[:, 1] if oy else 1 - fr_[:, 1]
            ii, jj = base[:, 0] + ox, base[:, 1] + oy
            ok = (ii >= 0) & (ii < RES) & (jj >= 0) & (jj < RES)
            img += np.bincount((ii * RES + jj)[ok], weights=(wx * wy)[ok], minlength=RES * RES)
    return img


TGT_A = [1.0 - np.exp(-1.5 * counts(tgt, th, ph)) for th, ph in VIEWS]

# the pieces from the detector's tracks
pieces = []
cams = {}
CLS = {}
for name, tj, fdir, az in views_in:
    tr = json.load(open(tj))
    cam = VA.side_camera(tgt) if float(az) == 90.0 else None
    if cam is None:                                            # the front camera: azimuth az
        center = tgt.mean(0)
        radius = float(np.linalg.norm(tgt - center, axis=1).max())
        W, H, AZ, EL = 3840, 2160, math.radians(float(az)), math.radians(18.0)
        cp = center + 3.6 * radius * np.array([math.cos(EL) * math.sin(AZ), math.sin(EL), math.cos(EL) * math.cos(AZ)])
        fwd = (center - cp) / np.linalg.norm(center - cp)
        right = np.cross([0.0, 1.0, 0.0], fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        tan_y = math.tan(math.radians(15.0))
        tan_x = tan_y * W / H

        def pixels(p, cp=cp, right=right, up=up, fwd=fwd, tan_x=tan_x, tan_y=tan_y, W=W, H=H):
            d = p - cp
            xc, yc, zc = d @ right, d @ up, d @ fwd
            return np.stack((((xc / zc) / tan_x + 1) / 2 * W - 0.5, (1 - (yc / zc) / tan_y) / 2 * H - 0.5), 1)
        cam = (pixels, (lambda p, cp=cp, fwd=fwd: (p - cp) @ fwd), (lambda zc, W=W, tan_x=tan_x: W / (2.0 * zc * tan_x)))
    cams[name] = (cam, fdir)
    for tid, t in tr.items():
        if t["cls_i"] <= 0 or t["max_px"] >= 5000:
            continue
        k = int(t["k_max"])
        x = np.asarray(fr[k], np.float64)
        key = (name, k)
        if key not in CLS:
            CLS[key] = VA.classify(x, a, cam)
        ap, kind, height, occl = CLS[key]
        ap_set = set(ap.tolist())
        if not (kind == 2).any():
            continue
        x0b, y0b, x1b, y1b = t["bbox"]
        q = cam[0](x[ap])
        inb = (q[:, 0] >= x0b - 10) & (q[:, 0] <= x1b + 10) & (q[:, 1] >= y0b - 10) & (q[:, 1] <= y1b + 10)
        sel = ap[inb & (kind == 2) & ~occl]
        if not len(sel):
            continue
        # the blobs' connected sets
        tree = cKDTree(x)
        members = set(sel.tolist())
        frontier = list(sel)
        while frontier:
            nxt = []
            for p in frontier:
                for j in tree.query_ball_point(x[p], 1.5 * a):
                    if j not in members and j in ap_set:
                        members.add(j)
                        nxt.append(j)
            frontier = nxt
        pieces.append(dict(view=name, track=int(tid), t0=float(t["t_first"]), t1=float(t["t_last"]), k=k,
                           bbox=t["bbox"], particles=sorted(members)))
# merge pieces sharing particles
merged = []
for p in sorted(pieces, key=lambda d: d["t0"]):
    for m in merged:
        if set(p["particles"]) & set(m["particles"]):
            m["particles"] = sorted(set(m["particles"]) | set(p["particles"]))
            m["t0"], m["t1"] = min(m["t0"], p["t0"]), max(m["t1"], p["t1"])
            m["tracks"].append((p["view"], p["track"]))
            break
    else:
        merged.append(dict(p, tracks=[(p["view"], p["track"])]))
print(f"[{label}] visible floating pieces: {len(merged)} (from {len(pieces)} confirmed tracks)", flush=True)

# per piece and window
xe = np.asarray(fr[-1], np.float64)
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
pe = cKDTree(xe).query_pairs(1.5 * a, output_type="ndarray")
ge = coo_matrix((np.ones(len(pe), np.int8), (pe[:, 0], pe[:, 1])), shape=(len(xe), len(xe)))
_, cle = connected_components(ge, directed=False)
body_end = cle == np.bincount(cle).argmax()
cache_disp = {}
rows_out = []
out = open(out_prefix + ".jsonl", "w")
for pi, m in enumerate(merged):
    P = np.asarray(m["particles"], np.int64)
    fate = float(body_end[P].mean())
    k_or = int(np.argmin(np.abs(raws / 240.0 - (m["t0"] - 0.25))))
    t_or = tgt[ttree.query(np.asarray(fr[k_or], np.float64)[P])[1]]
    print(f"\n== {label} piece {pi}: {m['tracks']} {m['t0']:.2f}-{m['t1']:.2f} s, site {m['bbox']}, {len(P)} particles "
          f"{P.tolist()[:10]}{' ...' if len(P) > 10 else ''}; at the end in the body {fate:.2f}", flush=True)
    w0, w1 = int(m["t0"] * 6), int(m["t1"] * 6)
    ws = list(range(w0, w1 + 1))
    if len(ws) > 10:
        ws = ws[::int(math.ceil(len(ws) / 10))]
    for w in ws:
        k = int(np.argmin(np.abs(raws - 40 * w)))
        x = np.asarray(fr[k], np.float64)
        xp = x[P]
        sd = sdist(xp)
        dv, dg = dt_at(xp)
        gate = VA.gate_values(x)[P] if hasattr(VA, "gate_values") else None
        if gate is None:
            dd, _ = cKDTree(x).query(x, 9, workers=16)
            r = dd[:, 8] / np.median(dd[:, 8])
            gate = np.clip((r - 1.2) / 0.6, 0, 1)[P]
        dn, it = ttree.query(xp)
        elig = (dn > berth) & (dn < ldx)
        own = np.array([inside_segment(t_or[i], tgt[it[i]]) for i in range(len(P))])
        if k not in cache_disp:
            cache_disp[k] = transport(x)
        disp = cache_disp[k]
        dm = np.linalg.norm(disp, axis=1) / a
        tdir = tgt[it] - xp
        cosd = (disp[P] * tdir).sum(1) / (np.linalg.norm(disp[P], axis=1) * np.linalg.norm(tdir, axis=1) + 1e-12)
        body_med = float(np.median(dm[np.random.default_rng(w).choice(len(x), 5000, replace=False)]))
        seen, rise = 0, 0.0
        mask = np.ones(len(x), bool)
        mask[P] = False
        for vi, (th, ph) in enumerate(VIEWS):
            cb = counts(x[mask], th, ph)
            cp = counts(xp, th, ph)
            dA = (1 - np.exp(-1.5 * (cb + cp))) - (1 - np.exp(-1.5 * cb))
            rise = max(rise, float(dA.max()))
            seen += int(((dA > 0.01) & (TGT_A[vi] < 0.5)).any())
        row = dict(run=label, piece=pi, window=w, t=w / 6.0, n=int(len(P)), sd_med=float(np.median(sd)),
                   sd_max=float(sd.max()), dt_med=float(np.median(dv)), dt_zero=float((dv <= 1e-9).mean()),
                   dt_grad_med=float(np.median(dg)), gate_on=float((gate > 0).mean()), nn_dist_med=float(np.median(dn / a)),
                   nn_elig=float(elig.mean()), nn_own=float(own.mean()), disp_med=float(np.median(dm[P])),
                   disp_body=body_med, disp_cos=float(np.median(cosd)), views_seen=seen, alpha_rise=rise, fate=fate)
        out.write(json.dumps(row) + "\n")
        rows_out.append(row)
        print(f"  w{w} ({w / 6:.2f} s): mesh distance median {row['sd_med']:+.2f} (max {row['sd_max']:+.2f}) a; cleanup "
              f"field {row['dt_med']:.2f} a, zero at {row['dt_zero']:.2f}, |grad| {row['dt_grad_med']:.2f}; gate on "
              f"{row['gate_on']:.2f}; near band: nearest target {row['nn_dist_med']:.2f} a, eligible {row['nn_elig']:.2f},"
              f" own side {row['nn_own']:.2f}; transport |d| {row['disp_med']:.2f} a (body {body_med:.2f}), cos to the "
              f"nearest target {row['disp_cos']:+.2f}; training views that can see it {seen} of {len(VIEWS)} (alpha "
              f"rise {rise:.3f})", flush=True)
out.close()
if rows_out:
    by = {}
    for r in rows_out:
        by.setdefault(r["piece"], []).append(r)
    blind = sum(any(r["sd_med"] > 1.0 and r["dt_zero"] >= 0.5 for r in rs) for rs in by.values())
    unseen = sum(all(r["views_seen"] == 0 for r in rs) for rs in by.values())
    print(f"\n== {label}: {len(by)} pieces; outside the mesh by more than a pitch while the cleanup field reads zero (at "
          f"some window): {blind} ({blind / len(by):.2f}); invisible to every training view at every window: {unseen} "
          f"({unseen / len(by):.2f}); piece-windows {len(rows_out)}: mesh > 1 a {np.mean([r['sd_med'] > 1 for r in rows_out]):.2f}, "
          f"cleanup zero {np.mean([r['dt_zero'] >= 0.5 for r in rows_out]):.2f}, both "
          f"{np.mean([(r['sd_med'] > 1) and (r['dt_zero'] >= 0.5) for r in rows_out]):.2f}; gate on "
          f"{np.mean([r['gate_on'] for r in rows_out]):.2f}; near band eligible {np.mean([r['nn_elig'] for r in rows_out]):.2f}",
          flush=True)

# contact sheet
from PIL import Image, ImageDraw  # noqa: E402
tiles = []
for pi, m in enumerate(merged[:40]):
    cam, fdir = cams[m["view"]]
    try:
        im = Image.open(os.path.join(fdir, f"{m['k']:04d}.jpg")).convert("RGB")
    except OSError:
        continue
    x0b, y0b, x1b, y1b = m["bbox"]
    cx, cy = (x0b + x1b) // 2, (y0b + y1b) // 2
    half = max(80, (x1b - x0b), (y1b - y0b))
    crop = im.crop((cx - half, cy - half, cx + half, cy + half)).resize((240, 240))
    ImageDraw.Draw(crop).text((4, 4), f"{label} #{pi} {m['view']} {m['t0']:.2f}-{m['t1']:.2f}s n{len(m['particles'])}",
                              fill=(0, 0, 0))
    tiles.append(crop)
if tiles:
    sheet = Image.new("RGB", (240 * 8, 240 * ((len(tiles) + 7) // 8)), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 8) * 240, (i // 8) * 240))
    sheet.save(out_prefix + "_sheet.jpg", quality=90)
    print(f"[{label}] sheet {out_prefix}_sheet.jpg ({len(tiles)} tiles)", flush=True)
