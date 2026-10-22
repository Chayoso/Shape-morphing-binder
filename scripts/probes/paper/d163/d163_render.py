"""d163_render.py -- D163's deliverable renders (CPU, Embree via d157_lib): D161's studio look (d161_render.py: ambient,
three GGX lights fixed to the camera, the grey gradient backdrop, the tone map, sRGB, a light-grey body) with the key
light grazing the faced surface (`rake` degrees from the view axis, from the viewer's upper left), on rectangular frames.

The camera is right-handed (right = forward x up): the true view, in which the plaque's words read correctly. D161's
camera (right = up x forward) draws a mirror image; this one equals it flipped back left-right (the hflip the studio
raster's videos get at encoding), so no flip is applied here. Surfaces: a mesh in the run's frame (the smooth or relief
target), or a state's display surface (the particles' Zhu-Bridson field with Solenthaler et al.'s correction, D157's
field_mesh "zbs", at the run's pitch).

  mesh    OUT.png OBJ BASE N AZ EL W H [RAKE] [ZOOM] [LOOK|-]
  video   OUT_DIR BASE N TOP_NPZ BOTTOM_NPZ TOP_LABEL BOTTOM_LABEL [W] [H] [AZ] [EL] [ORBIT] [WORKERS] [RAKE]
          frames OUT_DIR/f%05d.jpg at W x H: TOP above BOTTOM (each W x H/2), every kept frame of the longer run (a shorter
          run's last frame held, marked "ended"), then 20 frames held, then ORBIT frames turning both end states once
          around the vertical axis (both faces); the time and the arm's name on each panel.
  closeup OUT_DIR BASE N RELIEF_OBJ NPZ:LABEL [NPZ:LABEL ...] [W] [H]
          the end state of each run and the relief mesh, each word (front: +z, back: -z) seen from a raking three-quarter
          view, zoomed on the word's half: OUT_DIR/<label>_<face>_<half>.png and a sheet OUT_DIR/closeup_sheet.jpg
  sheet   OUT.jpg BASE N RELIEF_OBJ TIMES NPZ:LABEL [NPZ:LABEL ...]
          a contact sheet: one row per run at the kept frames nearest TIMES (s, comma-separated) and its end, the relief
          mesh (the render's target) at the row's end."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d161"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d157_lib as D7  # noqa: E402
from d161_lib import target_frame  # noqa: E402

ALBEDO = np.array((.74, .75, .76))


def camera(look, dist, view, W, H, tan_h):
    cam = look + dist * D7.dvec(*view)
    fwd = D7.unit(look - cam)
    right = D7.unit(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
    up = np.cross(right, fwd)
    tan_v = tan_h * H / W
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    rays = ((2 * (xx + .5) / W - 1)[..., None] * tan_h * right + (1 - 2 * (yy + .5) / H)[..., None] * tan_v * up + fwd)
    return cam, D7.unit(rays), right, up, fwd


def render_rake(surf, look, dist, view, W, H, tan_h, rake=72.0, key=3.0):
    cam, dirs, right, up, fwd = camera(look, dist, view, W, H, tan_h)
    rays = np.concatenate([np.broadcast_to(cam, dirs.shape).reshape(-1, 3), dirs.reshape(-1, 3)], 1)
    hit, pid, uv, _ = surf.cast(rays)
    n = np.zeros((len(rays), 3))
    n[hit] = surf.normals_at(pid[hit], uv[hit])
    cov = hit.reshape(H, W).astype(np.float64)
    ns = D7.unit(n.reshape(H, W, 3) + 1e-12)
    v = -dirs
    nv = np.clip((ns * v).sum(-1), 1e-4, None)
    amb = .14 + .08 * (.5 + .5 * ns[..., 1])
    rad = ALBEDO * amb[..., None]
    rough = .42
    a2 = rough ** 4
    k = (rough + 1) ** 2 / 8
    vis_v = nv / (nv * (1 - k) + k)
    r = np.radians(rake)
    key_dir = np.cos(r) * (-fwd) + np.sin(r) * D7.unit(-0.92 * right + 0.38 * up)    # grazing, from the viewer's upper left
    lights = ((key_dir, (1., .95, .88), key), (D7.unit(.8 * right + .15 * up - .7 * fwd), (.73, .85, 1.), .35),
              (D7.unit(.4 * right + .75 * up + .8 * fwd), (.85, .92, 1.), .9))
    for l, col, inten in lights:
        l = D7.unit(np.asarray(l, np.float64))
        h = D7.unit(v + l)
        nl = np.clip((ns * l).sum(-1), 0, 1)
        nh = np.clip((ns * h).sum(-1), 0, 1)
        vh = np.clip((v * h).sum(-1), 0, 1)
        Dd = a2 / (np.pi * np.maximum((nh * nh * (a2 - 1) + 1) ** 2, 1e-8))
        Fr = .04 + .96 * (1 - vh) ** 5
        vis_l = nl / (nl * (1 - k) + k)
        spec = Dd * Fr * vis_v * vis_l / np.maximum(4 * nv * nl, 1e-5)
        diff = ALBEDO * (1 - Fr[..., None]) / np.pi
        rad = rad + (diff + spec[..., None]) * (np.array(col) * inten) * nl[..., None]
    blend = (np.arange(H) / max(1, H - 1))[:, None, None]
    bg = (1 - blend) * np.array((.30, .30, .31)) + blend * np.array((.56, .56, .57))
    lin = rad * cov[..., None] + bg * (1 - cov[..., None])
    mapped = lin / (1 + lin)
    srgb = np.clip(np.where(mapped <= .0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1 / 2.4) - .055), 0, 1)
    return (srgb * 255 + .5).astype(np.uint8)


def mesh_surf(obj, frame):
    import trimesh
    m = trimesh.load(obj, process=False, force="mesh")            # d163 meshes: orientation 'id'
    V = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    F = np.asarray(m.faces, np.int64)
    return D7.Surf(V, F, cn=D7.corner_normals(V, F, crease=60.0))


def label(arr, text, sub=None, scale=34):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", max(14, arr.shape[0] // scale))
    except OSError:
        f = ImageFont.load_default()
    d.text((int(arr.shape[0] / 40), int(arr.shape[0] / 50)), text, fill=(18, 18, 18), font=f)
    if sub:
        d.text((int(arr.shape[0] / 40), int(arr.shape[0] / 50 + 1.4 * f.size)), sub, fill=(35, 35, 35), font=f)
    return np.asarray(im)


def setup(base, n, seed=97):
    frame, x_t = target_frame("assets/isosphere.obj", base, n, seed)
    c = x_t.astype(np.float64).mean(0)
    rad = float(np.linalg.norm(x_t - c, axis=1).max())
    a = .708 * D7.median_kth_spacing(x_t.astype(np.float64), 8, 20000)
    return frame, x_t, c, rad, a


def seed_of(path):
    return 99 if "_99" in Path(path).name else 97


def display(X, a):
    surf, _ = D7.field_mesh("zbs", np.asarray(X, np.float64), a)
    return surf


# ---------------------------------------------------------------------------------------------------------------- video
_V = {}


def _end_arrays(args):
    X, a = args
    s = display(X, a)
    return s.V, s.F, s.vn


def _video_job(args):
    k, kind, payload = args
    g = _V
    W, H, view, rake = g["W"], g["H"], g["view"], g["rake"]
    Hp = H // 2
    panels = []
    for side in ("top", "bottom"):
        F, raws, lab = g[side]
        if kind == "morph":
            r = payload
            j = int(np.searchsorted(raws, r)) if r in raws else len(raws) - 1
            ended = r not in raws
            t = raws[j] / 240.0
            surf = g[f"end_{side}"] if ended else display(F[j], g["a"])     # an ended run: its end, computed once
            vw = view
            sub = f"t = {t:.2f} s" + (" (ended, held)" if ended else "")
        else:
            j, ang = len(raws) - 1, payload
            surf = g[f"end_{side}"]                          # computed once before the fork
            vw = (view[0] + ang, view[1])
            sub = f"end (t = {raws[j] / 240.0:.2f} s), turned {np.degrees(ang):.0f} deg"
        img = render_rake(surf, g["look"], g["dist"], vw, W, Hp, g["tan_h"], rake=rake)
        panels.append(label(img, lab, sub, scale=22))
    from PIL import Image
    Image.fromarray(np.concatenate(panels, 0)).save(Path(g["out"]) / f"f{k:05d}.jpg", quality=93)
    return k


def video(argv):
    out = Path(argv[0])
    base, n, top, bottom, tl, bl = argv[1], int(argv[2]), argv[3], argv[4], argv[5], argv[6]
    W = int(argv[7]) if len(argv) > 7 else 3840
    H = int(argv[8]) if len(argv) > 8 else 2160
    az = float(argv[9]) if len(argv) > 9 else 0.30
    el = float(argv[10]) if len(argv) > 10 else 0.18
    orbit = int(argv[11]) if len(argv) > 11 else 160
    workers = int(argv[12]) if len(argv) > 12 else 12
    rake = float(argv[13]) if len(argv) > 13 else 72.0
    out.mkdir(parents=True, exist_ok=True)
    frame, x_t, c, rad, a = setup(base, n, seed_of(top))
    zt, zb = np.load(top, allow_pickle=True), np.load(bottom, allow_pickle=True)
    _V.update(W=W, H=H, view=(az, el), rake=rake, a=a, look=c, dist=3.6 * rad, tan_h=1.8 * rad / (3.6 * rad),
              out=str(out), top=(zt["frames"], np.asarray(zt["raws"]), tl), bottom=(zb["frames"], np.asarray(zb["raws"]), bl))
    T = sorted(set(np.asarray(zt["raws"]).tolist()) | set(np.asarray(zb["raws"]).tolist()))
    cap = float(os.environ.get("D163_CAP_S", "0"))            # >0: the morph shown to this time, then the ends
    if cap > 0:
        T = [r for r in T if r / 240.0 <= cap]
    # the two end surfaces once, in a child (torch's OpenMP pool must not start in the parent before the frame pool forks);
    # the parent builds their Embree scenes, which the forked workers share
    from multiprocessing import get_context
    with get_context("fork").Pool(2) as pool:
        ends = pool.map(_end_arrays, [(np.asarray(_V["top"][0][-1]), a), (np.asarray(_V["bottom"][0][-1]), a)])
    _V["end_top"], _V["end_bottom"] = (D7.Surf(V, F, vn=vn) for V, F, vn in ends)
    jobs =[(k, "morph", r) for k, r in enumerate(T)]
    k0 = len(jobs)
    jobs += [(k0 + i, "orbit", 0.0) for i in range(20)]
    jobs += [(k0 + 20 + i, "orbit", 2 * np.pi * (i + 1) / orbit) for i in range(orbit)]
    todo = [j for j in jobs if not (out / f"f{j[0]:05d}.jpg").exists()]
    from multiprocessing import get_context
    morph = [j for j in todo if j[1] == "morph"]
    orb = [j for j in todo if j[1] == "orbit"]
    with get_context("fork").Pool(workers) as pool:
        for k in pool.imap_unordered(_video_job, morph, chunksize=1):
            print("frame", k, flush=True)
    if orb:
        with get_context("fork").Pool(workers) as pool:
            for k in pool.imap_unordered(_video_job, orb, chunksize=2):
                print("orbit", k, flush=True)
    print("frames done", len(jobs), flush=True)


# -------------------------------------------------------------------------------------------------------------- closeup
def closeup(argv):
    out = Path(argv[0])
    base, n, relief = argv[1], int(argv[2]), argv[3]
    items = [s for s in argv[4:] if ":" in s]
    rest = [s for s in argv[4:] if ":" not in s]
    W = int(rest[0]) if rest else 1920
    H = int(rest[1]) if len(rest) > 1 else 1080
    out.mkdir(parents=True, exist_ok=True)
    frame, x_t, c, rad, a = setup(base, n, seed_of(items[0].split(":")[0]) if items else 97)
    surfs = [("render target (relief mesh)", mesh_surf(relief, frame))]
    for it in items:
        p, lab = it.split(":", 1)
        z = np.load(p, allow_pickle=True)
        surfs.append((lab, display(np.asarray(z["frames"][-1], np.float64), a)))
    half = 0.5 * (x_t[:, 0].max() - x_t[:, 0].min())
    tiles = []
    for face, az in (("front", 0.0), ("back", np.pi)):
        for hname, sx in (("left", -0.5), ("right", 0.5)):
            # the half of the word on the viewer's left / right (the back face's viewer has -x on the right)
            xw = sx * half * (1 if face == "front" else -1)
            look = c + np.array([xw, 0.0, 0.0])
            view = (az + 0.30, 0.20)
            row = []
            for lab, s in surfs:
                img = render_rake(s, look, 1.6 * rad, view, W, H, 0.55 * half / (1.6 * rad), rake=74.0)
                from PIL import Image
                Image.fromarray(img).save(out / f"{lab.split(' (')[0].replace(' ', '_').replace('+', '')}_{face}_{hname}.png")
                row.append(label(img, lab, f"{face} word, {hname} half (end state)", scale=20))
            tiles.append(np.concatenate(row, 1))
    from PIL import Image
    sheet = np.concatenate(tiles, 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / "closeup_sheet.jpg", quality=92)


def sheet(argv):
    out, base, n, relief, times = argv[0], argv[1], int(argv[2]), argv[3], [float(t) for t in argv[4].split(",")]
    items = argv[5:]
    frame, x_t, c, rad, a = setup(base, n, seed_of(items[0].split(":")[0]))
    W, H = 960, 400
    view, dist, tan_h = (0.30, 0.18), 3.6 * rad, 1.25 * rad / (3.6 * rad)
    tgt = label(render_rake(mesh_surf(relief, frame), c, dist, view, W, H, tan_h), "render target (relief mesh)",
                "the 3D target is the smooth plaque", scale=16)
    rows = []
    for it in items:
        p, lab = it.split(":", 1)
        z = np.load(p, allow_pickle=True)
        raws = np.asarray(z["raws"])
        js = sorted(set([int(np.argmin(np.abs(raws / 240.0 - t))) for t in times] + [len(raws) - 1]))
        row = []
        for j in js:
            img = render_rake(display(z["frames"][j], a), c, dist, view, W, H, tan_h)
            row.append(label(img, lab, f"t = {raws[j] / 240.0:.2f} s" + (" (end)" if j == len(raws) - 1 else ""), scale=16))
        row.append(tgt)
        rows.append(np.concatenate(row, 1))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=128) for r in rows]
    from PIL import Image
    Image.fromarray(np.concatenate(rows, 0)).save(out, quality=92)


def main():
    mode = sys.argv[1]
    if mode == "mesh":
        out, obj, base, n, az, el, W, H = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), \
            float(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
        rake = float(sys.argv[10]) if len(sys.argv) > 10 else 72.0
        zoom = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
        frame, x_t, c, rad, a = setup(base, n)
        look = c if len(sys.argv) <= 12 or sys.argv[12] == "-" else np.array([float(v) for v in sys.argv[12].split(",")])
        img = render_rake(mesh_surf(obj, frame), look, 3.6 * rad / zoom, (az, el), W, H, 1.25 * rad / (3.6 * rad), rake=rake)
        from PIL import Image
        Image.fromarray(img).save(out)
    elif mode == "video":
        video(sys.argv[2:])
    elif mode == "closeup":
        closeup(sys.argv[2:])
    elif mode == "sheet":
        sheet(sys.argv[2:])


if __name__ == "__main__":
    main()
