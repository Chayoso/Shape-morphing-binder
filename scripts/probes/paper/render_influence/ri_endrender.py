"""ri_endrender.py OUT_DIR DMAX MESH [MESH ...] -- render influence (2026-10-22; CPU): the end-surface map of
ri_endmap.py drawn in the studio look (D157's ray-cast studio model: light-grey body, three studio lights, the grey
gradient backdrop, tone map, sRGB). The body is the twin's display surface; its albedo is the studio grey where the
full arm's surface is within the run-to-run floor of the twin's (|delta| <= 2 sigma), else tinted toward red (the full
arm's surface further out) or blue (further in) by |delta| / DMAX pitches (one fixed scale for every mesh and view).
The studio camera mirrors left-right (its right vector is up x forward); the image is flipped back before saving."""
import sys
from pathlib import Path

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d156")
import numpy as np  # noqa: E402
import d157_lib as D7  # noqa: E402
from PIL import Image  # noqa: E402


def srgb_to_lin(h):
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


GREY = D7.ALBEDO
RED, BLUE = srgb_to_lin("#e34948"), srgb_to_lin("#2a78d6")


def colours(delta, sigma, ok, dmax):
    t = np.clip(delta / dmax, -1, 1)
    beyond = ok & (np.abs(delta) > 2 * sigma)
    pole = np.where((t > 0)[:, None], RED[None], BLUE[None])
    w = np.where(beyond, np.abs(t), 0.0)[:, None]
    return GREY[None] * (1 - w) + pole * w


def render(surf, vcol, center, radius, view, res=1000, ss=2):
    cam, dirs, right, up, fwd = D7.studio_cam(center, radius, view, res * ss)
    rays = np.concatenate([np.broadcast_to(cam, dirs.shape).reshape(-1, 3), dirs.reshape(-1, 3)], 1)
    hit, pid, uv, _ = surf.cast(rays)
    n = np.zeros((len(rays), 3))
    alb = np.zeros((len(rays), 3))
    n[hit] = surf.normals_at(pid[hit], uv[hit])
    w = np.concatenate([1 - uv[hit, :1] - uv[hit, 1:], uv[hit]], 1)
    alb[hit] = (w[:, :, None] * vcol[surf.F[pid[hit]]]).sum(1)
    R = res * ss
    hit, n, alb = hit.reshape(R, R), n.reshape(R, R, 3), alb.reshape(R, R, 3)
    cov = hit.reshape(res, ss, res, ss).mean((1, 3))
    ns = D7.unit((n * hit[..., None]).reshape(res, ss, res, ss, 3).sum((1, 3)))
    A = (alb * hit[..., None]).reshape(res, ss, res, ss, 3).sum((1, 3)) / np.maximum(cov[..., None] * ss * ss, 1e-9)
    _, dirs1, _, _, _ = D7.studio_cam(center, radius, view, res)
    v = -dirs1
    nv = np.clip((ns * v).sum(-1), 1e-4, None)
    amb = .20 + .12 * (.5 + .5 * ns[..., 1])
    rad = A * amb[..., None]
    rough = .36
    a2 = rough ** 4
    k = (rough + 1) ** 2 / 8
    vis_v = nv / (nv * (1 - k) + k)
    for d, col, inten in D7.LIGHTS:
        l = D7.unit(d[0] * right + d[1] * up + d[2] * fwd)
        h = D7.unit(v + l)
        nl = np.clip((ns * l).sum(-1), 0, 1)
        nh = np.clip((ns * h).sum(-1), 0, 1)
        vh = np.clip((v * h).sum(-1), 0, 1)
        D = a2 / (np.pi * np.maximum((nh * nh * (a2 - 1) + 1) ** 2, 1e-8))
        Fr = .04 + .96 * (1 - vh) ** 5
        vis_l = nl / (nl * (1 - k) + k)
        spec = D * Fr * vis_v * vis_l / np.maximum(4 * nv * nl, 1e-5)
        diff = A * (1 - Fr[..., None]) / np.pi
        rad = rad + (diff + spec[..., None]) * (np.array(col) * inten) * nl[..., None]
    blend = (np.arange(res) / max(1, res - 1))[:, None, None]
    bg = (1 - blend) * np.array((.115, .137, .165)) + blend * np.array((.215, .237, .265))
    lin = rad * cov[..., None] + bg * (1 - cov[..., None])
    mapped = lin / (1 + lin)
    srgb = np.clip(np.where(mapped <= .0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1 / 2.4) - .055), 0, 1)
    return np.fliplr(srgb)                                        # undo the studio camera's mirror


def main():
    out, dmax, meshes = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
    views = {"front": (np.radians(35.0), np.radians(18.0)), "back": (np.radians(215.0), np.radians(18.0))}
    for mesh in meshes:
        z = np.load(out / f"endmap_{mesh}_vals.npz")
        V, F, n = z["V"].astype(np.float64), z["F"].astype(np.int64), z["n"].astype(np.float64)
        surf = D7.Surf(V, F, vn=n)
        vcol = colours(z["delta"].astype(np.float64), z["sigma"].astype(np.float64), z["ok"], dmax)
        plain = np.broadcast_to(GREY, vcol.shape).copy()
        for vn, view in views.items():
            for tag, col in (("map", vcol), ("plain", plain)):
                img = render(surf, col, np.asarray(z["center"], np.float64), float(z["radius"]), view)
                Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / f"endmap_{mesh}_{vn}_{tag}.png")
                print(mesh, vn, tag, flush=True)


if __name__ == "__main__":
    main()
