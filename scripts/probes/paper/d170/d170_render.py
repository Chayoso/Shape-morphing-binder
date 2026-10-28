"""d170_render.py -- D170's deliverable renders (CPU, Embree via d157_lib): D163's renderer unchanged (d163_render.render_rake:
D161's studio look, ambient + three GGX lights fixed to the camera, the grey gradient backdrop, the tone map, sRGB, a
light-grey body; the key light `rake` degrees from the view axis, from the viewer's upper left). Its camera is the true
(right-handed) view: it equals the studio raster's image flipped left-right, so no flip is applied at encoding.
Surfaces: a state's display surface (zbs, D157's field_mesh at the run's pitch) or a mesh put in the run's frame.

  sbs    OUT_DIR BASE N TWIN_NPZ FULL_NPZ STOP_RAW [W] [H] [AZ] [EL] [ORBIT] [WORKERS] [RAKE] [ZOOM] [LOOK|-]
         twin | full side by side (each W/2 x H) at every kept frame of either run (an ended run held), the full arm's
         panel marked once it is in its settle (raw > STOP_RAW: zero control, relaxation off); 20 frames held; ORBIT
         frames turning both end states once. OUT_DIR/f%05d.jpg
  close  OUT_DIR BASE N RELIEF_OBJ TWIN_NPZ FULL_NPZ STOP_RAW LOOK|auto|plaque VIEW|auto ZOOM [RAKE] [W] [H]
         raking close-ups (RAKE, default 84 deg) of the render target mesh, the twin's end, the full arm at its stop
         (before the settle) and at its end, one camera; LOOK auto: the centroid of the relief mesh's vertices with the
         pattern P > 0.5 (D161's .pattern.npz), VIEW auto: their mean normal; plaque: D163's halves of the front word.
         OUT_DIR/<tile>.png and OUT_DIR/closeup_sheet.jpg
  still  OUT.png BASE N NPZ IDX AZ EL [W] [H] [RAKE] [ZOOM]   one kept frame (IDX, -1 = the end) in the studio look"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d163")
import d157_lib as D7  # noqa: E402
from d163_render import display, label, render_rake, setup  # noqa: E402

TWIN_L, FULL_L = "dF_c + layer u (twin)", "dF_c + layer u + render loss (full)"


def mesh_surf(obj, frame):
    """A mesh in the run's frame, oriented as the sampler orients it (physmorph.sampling.mesh._oriented)."""
    from physmorph.sampling.mesh import _oriented
    m = _oriented(obj)
    V = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    F = np.asarray(m.faces, np.int64)
    return D7.Surf(V, F, cn=D7.corner_normals(V, F, crease=60.0))


def view_of(n):
    n = n / np.linalg.norm(n)
    return float(np.arctan2(n[0], n[2])), float(np.arcsin(np.clip(n[1], -1, 1)))


def seed_of(path):
    return 99 if "_99" in Path(path).name else 97


# ----------------------------------------------------------------------------------------------------------------- sbs
_V = {}


def _ends(args):
    X, a = args
    s = display(X, a)
    return s.V, s.F, s.vn


def _frame(args):
    k, kind, payload = args
    g = _V
    W, H, view, rake = g["W"], g["H"], g["view"], g["rake"]
    panels = []
    for side in ("twin", "full"):
        F, raws, lab = g[side]
        if kind == "morph":
            r = payload
            ended = r not in raws
            j = int(np.searchsorted(raws, r)) if not ended else len(raws) - 1
            surf = g[f"end_{side}"] if ended else display(F[j], g["a"])
            vw = view
            sub = f"t = {r / 240.0:.2f} s"
            if side == "full" and r > g["stop"]:
                sub += "  settle: zero control, relaxation off"
            if ended:
                sub = f"t = {raws[-1] / 240.0:.2f} s (ended, held)"
        else:
            surf, vw = g[f"end_{side}"], (view[0] + payload, view[1])
            sub = f"end (t = {raws[-1] / 240.0:.2f} s), turned {np.degrees(payload):.0f} deg"
        img = render_rake(surf, g["look"], g["dist"], vw, W // 2, H, g["tan_h"], rake=rake)
        panels.append(label(img, lab, sub, scale=26))
    from PIL import Image
    Image.fromarray(np.concatenate(panels, 1)).save(Path(g["out"]) / f"f{k:05d}.jpg", quality=93)
    return k


def sbs(argv):
    out = Path(argv[0])
    base, n, twin, full, stop = argv[1], int(argv[2]), argv[3], argv[4], int(argv[5])
    opt = argv[6:] + [None] * 10
    W, H = int(opt[0] or 3840), int(opt[1] or 2160)
    az, el = float(opt[2] or 0.30), float(opt[3] or 0.18)
    orbit, workers, rake = int(opt[4] or 160), int(opt[5] or 24), float(opt[6] or 72.0)
    zoom = float(opt[7] or 1.0)
    out.mkdir(parents=True, exist_ok=True)
    frame, x_t, c, rad, a = setup(base, n, seed_of(full))
    look = c if opt[8] in (None, "-") else np.array([float(v) for v in opt[8].split(",")])
    zt, zf = np.load(twin, allow_pickle=True), np.load(full, allow_pickle=True)
    dist = 3.6 * rad / zoom
    _V.update(W=W, H=H, view=(az, el), rake=rake, a=a, look=look, dist=dist, tan_h=1.15 * rad / zoom / dist, out=str(out), stop=stop,
              twin=(zt["frames"], np.asarray(zt["raws"]), TWIN_L), full=(zf["frames"], np.asarray(zf["raws"]), FULL_L))
    T = sorted(set(np.asarray(zt["raws"]).tolist()) | set(np.asarray(zf["raws"]).tolist()))
    cap = float(os.environ.get("D170_CAP_S", "0"))
    if cap > 0:
        T = [r for r in T if r / 240.0 <= cap]
    from multiprocessing import get_context
    with get_context("fork").Pool(2) as pool:
        ends = pool.map(_ends, [(np.asarray(_V["twin"][0][-1]), a), (np.asarray(_V["full"][0][-1]), a)])
    _V["end_twin"], _V["end_full"] = (D7.Surf(V, F, vn=vn) for V, F, vn in ends)
    jobs = [(k, "morph", r) for k, r in enumerate(T)]
    k0 = len(jobs)
    jobs += [(k0 + i, "orbit", 0.0) for i in range(20)]
    jobs += [(k0 + 20 + i, "orbit", 2 * np.pi * (i + 1) / orbit) for i in range(orbit)]
    todo = [j for j in jobs if not (out / f"f{j[0]:05d}.jpg").exists()]
    for part in ([j for j in todo if j[1] == "morph"], [j for j in todo if j[1] == "orbit"]):
        if part:
            with get_context("fork").Pool(workers) as pool:
                for k in pool.imap_unordered(_frame, part, chunksize=1):
                    print("frame", k, flush=True)
    print("frames done", len(jobs), flush=True)


# --------------------------------------------------------------------------------------------------------------- close
def close(argv):
    out = Path(argv[0])
    base, n, relief, twin, full, stop = argv[1], int(argv[2]), argv[3], argv[4], argv[5], int(argv[6])
    look_s, view_s, zoom = argv[7], argv[8], float(argv[9])
    rake = float(argv[10]) if len(argv) > 10 else 84.0
    W = int(argv[11]) if len(argv) > 11 else 1920
    H = int(argv[12]) if len(argv) > 12 else 1080
    out.mkdir(parents=True, exist_ok=True)
    frame, x_t, c, rad, a = setup(base, n, seed_of(full))
    rs = mesh_surf(relief, frame)
    zt, zf = np.load(twin, allow_pickle=True), np.load(full, allow_pickle=True)
    raws_f = np.asarray(zf["raws"])
    js = int(np.clip(np.searchsorted(raws_f, stop), 0, len(raws_f) - 1))
    tiles = [("render_target", "render target (relief mesh)", rs),
             ("twin_end", f"{TWIN_L}: end (t = {np.asarray(zt['raws'])[-1] / 240:.2f} s)", display(np.asarray(zt["frames"][-1], np.float64), a)),
             ("full_stop", f"{FULL_L}: stop (t = {raws_f[js] / 240:.2f} s)", display(np.asarray(zf["frames"][js], np.float64), a)),
             ("full_end", f"{FULL_L}: end after the settle (t = {raws_f[-1] / 240:.2f} s)", display(np.asarray(zf["frames"][-1], np.float64), a))]
    cams = []
    if look_s == "plaque":
        half = 0.5 * (x_t[:, 0].max() - x_t[:, 0].min())
        for hname, sx in (("left", -0.5), ("right", 0.5)):
            cams.append((f"front_{hname}", c + np.array([sx * half, 0.0, 0.0]), (0.30, 0.20), 1.6 * rad, 0.55 * half / (1.6 * rad)))
    else:
        if look_s == "auto" or view_s == "auto":
            from d161_pattern import Pattern
            pat = Pattern.load(relief + ".pattern.npz")
            V = rs.V
            vn = rs.vn if getattr(rs, "vn", None) is not None else None
            if vn is None:
                import trimesh
                vn = np.asarray(trimesh.Trimesh(V, rs.F, process=False).vertex_normals)
            sub = np.random.default_rng(0).choice(len(V), min(400000, len(V)), replace=False)
            P = pat.at(V[sub], vn[sub], zpts=V[sub])
            hot = sub[P > 0.5]
            look = V[hot].mean(0) if look_s == "auto" else np.array([float(v) for v in look_s.split(",")])
            view = view_of(vn[hot].mean(0)) if view_s == "auto" else tuple(float(v) for v in view_s.split(","))
            ext = float(np.linalg.norm(V[hot] - look, axis=1).max())
        elif look_s.startswith("rel:"):                      # the body's centre + radius x (x, y, z), D159's close-up looks
            look = c + rad * np.array([float(v) for v in look_s[4:].split(",")])
            view = tuple(float(v) for v in view_s.split(","))
            ext = rad / zoom
        else:
            look = np.array([float(v) for v in look_s.split(",")])
            view = tuple(float(v) for v in view_s.split(","))
            ext = rad / zoom
        if look_s == "auto":
            view = (view[0] + 0.25, max(view[1], -1.2))       # a three-quarter turn off the face, as D163's close-ups
        dist = 1.6 * rad
        cams.append(("face", look, view, dist, 1.15 * ext / dist))
    from PIL import Image
    rows = []
    for cname, look, view, dist, tan_h in cams:
        row = []
        for key, lab, s in tiles:
            img = render_rake(s, look, dist, view, W, H, tan_h, rake=rake)
            Image.fromarray(img).save(out / f"{key}_{cname}_rake{rake:.0f}.png")
            row.append(label(img, lab, f"raking light {rake:.0f} deg", scale=20))
        rows.append(np.concatenate([np.concatenate(row[:2], 1), np.concatenate(row[2:], 1)], 0))
    sheet = np.concatenate(rows, 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / f"closeup_sheet_rake{rake:.0f}.jpg", quality=92)
    print("close done", out, flush=True)


def still(argv):
    out, base, n, npz, idx, az, el = argv[0], argv[1], int(argv[2]), argv[3], int(argv[4]), float(argv[5]), float(argv[6])
    W = int(argv[7]) if len(argv) > 7 else 3840
    H = int(argv[8]) if len(argv) > 8 else 2160
    rake = float(argv[9]) if len(argv) > 9 else 72.0
    zoom = float(argv[10]) if len(argv) > 10 else 1.0
    frame, x_t, c, rad, a = setup(base, n, seed_of(npz))
    z = np.load(npz, allow_pickle=True)
    s = display(np.asarray(z["frames"][idx], np.float64), a)
    dist = 3.6 * rad / zoom
    img = render_rake(s, c, dist, (az, el), W, H, 1.8 * rad / zoom / dist, rake=rake)
    from PIL import Image
    Image.fromarray(img).save(out)
    print("still done", out, flush=True)


if __name__ == "__main__":
    {"sbs": sbs, "close": close, "still": still}[sys.argv[1]](sys.argv[2:])
