"""d161_render.py -- D161: studio-look stills and side-by-side videos with a raking light (CPU, Embree via d157_lib).

The studio look of D157's reader (render/studio.py's model: ambient, three GGX lights fixed to the camera, the grey
gradient backdrop, the tone map, sRGB; a light-grey body) with the key light turned to graze the surface the camera
faces (`rake` degrees from the view axis, from the camera's left and a little above), so that a relief of a few
hundredths of a radian of slope reads. Surfaces: a mesh (the smooth or relief target, in the run's frame) or a state's
display surface (the particles' Zhu-Bridson field with Solenthaler et al.'s correction, D157's field_mesh "zbs"). The
camera looks at LOOK (default the body's centre) from 3.6 body radii / ZOOM along the view (THETA, PHI), 30 deg.

  mesh  OUT.png OBJ BASE N THETA PHI [RES] [RAKE] [ZOOM] [LOOKX,LOOKY,LOOKZ]
  pair  OUT_DIR FULL_NPZ TWIN_NPZ RELIEF_OBJ BASE N THETA PHI [RES] [RAKE] [ZOOM] [LOOK|-] [EVERY]
        every EVERY-th common kept frame: three panels, 'dF_c + layer u + render loss (full)' | 'dF_c + layer u' |
        'render target (relief mesh)', labelled, with the time; then OUT_DIR/pair.mp4 (ffmpeg, 12 fps)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d157_lib as D7  # noqa: E402
from d161_lib import target_frame  # noqa: E402

ALBEDO = np.array((.74, .75, .76))                    # light grey (the studio look)


def camera(look, dist, view, res, fov=30.0):
    cam = look + dist * D7.dvec(*view)
    fwd = D7.unit(look - cam)
    right = D7.unit(np.cross(np.array([0.0, 1.0, 0.0]), fwd))
    up = np.cross(fwd, right)
    tan = np.tan(np.radians(fov) / 2)
    yy, xx = np.meshgrid(np.arange(res, dtype=np.float64), np.arange(res, dtype=np.float64), indexing="ij")
    rays = ((2 * (xx + .5) / res - 1)[..., None] * tan * right + (1 - 2 * (yy + .5) / res)[..., None] * tan * up + fwd)
    return cam, D7.unit(rays), right, up, fwd


def render_rake(surf, look, dist, view, res=768, ss=2, rake=72.0, key=3.0):
    cam, dirs, right, up, fwd = camera(look, dist, view, res * ss)
    rays = np.concatenate([np.broadcast_to(cam, dirs.shape).reshape(-1, 3), dirs.reshape(-1, 3)], 1)
    hit, pid, uv, _ = surf.cast(rays)
    n = np.zeros((len(rays), 3))
    n[hit] = surf.normals_at(pid[hit], uv[hit])
    R = res * ss
    hit, n = hit.reshape(R, R), n.reshape(R, R, 3)
    cov = hit.reshape(res, ss, res, ss).mean((1, 3))
    ns = D7.unit((n * hit[..., None]).reshape(res, ss, res, ss, 3).sum((1, 3)))
    _, dirs1, _, _, _ = camera(look, dist, view, res)
    v = -dirs1
    nv = np.clip((ns * v).sum(-1), 1e-4, None)
    amb = .14 + .08 * (.5 + .5 * ns[..., 1])
    rad = ALBEDO * amb[..., None]
    rough = .42
    a2 = rough ** 4
    k = (rough + 1) ** 2 / 8
    vis_v = nv / (nv * (1 - k) + k)
    r = np.radians(rake)
    key_dir = np.cos(r) * (-fwd) + np.sin(r) * D7.unit(-0.92 * right + 0.38 * up)    # grazing, from the left, above
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
    blend = (np.arange(res) / max(1, res - 1))[:, None, None]
    bg = (1 - blend) * np.array((.30, .30, .31)) + blend * np.array((.56, .56, .57))      # grey gradient backdrop
    lin = rad * cov[..., None] + bg * (1 - cov[..., None])
    mapped = lin / (1 + lin)
    srgb = np.clip(np.where(mapped <= .0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1 / 2.4) - .055), 0, 1)
    return srgb.astype(np.float32), cov


def mesh_surf(obj, frame):
    from physmorph.sampling.mesh import _oriented
    m = _oriented(obj)
    V = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    F = np.asarray(m.faces, np.int64)
    return D7.Surf(V, F, cn=D7.corner_normals(V, F, crease=60.0))


def to_u8(img):
    return (img * 255 + .5).astype(np.uint8)


def label(arr, text, sub=None):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", max(14, arr.shape[0] // 34))
    except OSError:
        f = ImageFont.load_default()
    d.text((12, 10), text, fill=(20, 20, 20), font=f)
    if sub:
        d.text((12, 10 + int(1.4 * f.size)), sub, fill=(40, 40, 40), font=f)
    return np.asarray(im)


def setup(base, n, seed=97):
    frame, x_t = target_frame("assets/isosphere.obj", base, n, seed)
    c = x_t.astype(np.float64).mean(0)
    rad = float(np.linalg.norm(x_t - c, axis=1).max())
    a = .708 * D7.median_kth_spacing(x_t.astype(np.float64), 8, 20000)
    return frame, x_t, c, rad, a


def parse_look(s, c):
    return c if s in (None, "-") else np.array([float(v) for v in s.split(",")])


def main():
    mode = sys.argv[1]
    if mode == "mesh":
        out, obj, base, n, th, ph = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
        res = int(sys.argv[8]) if len(sys.argv) > 8 else 768
        rake = float(sys.argv[9]) if len(sys.argv) > 9 else 72.0
        zoom = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0
        frame, x_t, c, rad, a = setup(base, n)
        look = parse_look(sys.argv[11] if len(sys.argv) > 11 else None, c)
        img, _ = render_rake(mesh_surf(obj, frame), look, 3.6 * rad / zoom, (th, ph), res, rake=rake)
        from PIL import Image
        Image.fromarray(to_u8(img)).save(out)
    elif mode == "pair":
        out = Path(sys.argv[2])
        full, twin, relief, base, n = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7])
        th, ph = float(sys.argv[8]), float(sys.argv[9])
        res = int(sys.argv[10]) if len(sys.argv) > 10 else 512
        rake = float(sys.argv[11]) if len(sys.argv) > 11 else 72.0
        zoom = float(sys.argv[12]) if len(sys.argv) > 12 else 1.0
        every = int(sys.argv[14]) if len(sys.argv) > 14 else 1
        out.mkdir(parents=True, exist_ok=True)
        zf, zt = np.load(full, allow_pickle=True), np.load(twin, allow_pickle=True)
        seed = 99 if "_99" in Path(full).name or "99_" in Path(full).name else 97
        frame, x_t, c, rad, a = setup(base, n, seed)
        look = parse_look(sys.argv[13] if len(sys.argv) > 13 else None, c)
        dist = 3.6 * rad / zoom
        ref, _ = render_rake(mesh_surf(relief, frame), look, dist, (th, ph), res, rake=rake)
        ref = label(to_u8(ref), "render target (relief mesh)", "the 3D target stays smooth")
        rf, rt = list(np.asarray(zf["raws"])), list(np.asarray(zt["raws"]))
        Ff, Ft = zf["frames"], zt["frames"]
        # every common kept frame, then each run's own end held (a shorter run's last frame repeats)
        T = sorted(set(rf) | set(rt))
        T = T[::every] + ([T[-1]] if T[-1] not in T[::every] else [])
        last = {}
        for k, r in enumerate(T):
            panels = []
            for name, raws, F, tag in ((full, rf, Ff, "dF_c + layer u + render loss (full)"), (twin, rt, Ft, "dF_c + layer u")):
                j = raws.index(r) if r in raws else len(raws) - 1
                key = (name, j)
                if key not in last:
                    surf, _ = D7.field_mesh("zbs", np.asarray(F[j], np.float64), a)
                    img, _ = render_rake(surf, look, dist, (th, ph), res, rake=rake)
                    last = {kk: vv for kk, vv in last.items() if kk[0] != name}
                    last[key] = to_u8(img)
                t = raws[j] / 240.0
                panels.append(label(last[key], tag, f"t = {t:.2f} s" + ("" if r in raws else " (ended)")))
            panels.append(ref)
            from PIL import Image
            Image.fromarray(np.concatenate(panels, 1)).save(out / f"p{k:04d}.png")
            print(k, r, flush=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", str(out / "p%04d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "18",
                        str(out / "pair.mp4")], check=True)


if __name__ == "__main__":
    main()
