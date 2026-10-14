"""render_scene.py SCENE_NPZ OUT_DIR [+field=zb|zbs] [+camera=static|contact|follow] [+res=2160] [+az=0] [+el=12]
[+dist=1] [+smooth=0.75] [+stride=1] [+frames=a-b] [+floor=1] -- the two-body scene's frames (D147) in the studio look
(3840 x 2160 by default): each body's base display (the exterior's discs of its own particles, so that two bodies in
contact are not merged into one surface; the normals averaged twice over the reach as the deliverable's base display),
both drawn by one camera (framing: static by default, one shot for the whole run with the floor they land on; contact,
a still close-up of the rendered span (+frames); follow, the pair's box smoothed over +smooth seconds; +dist a zoom
factor) on the studio backdrop, and the floor as a plane in the backdrop's grey with a soft contact shadow (each body's
discs dropped onto the floor, weighted by their height above it and blurred). Frames in OUT_DIR/frames/NNNN.jpg (every
simulated frame by default; join at 240 / stride fps for real time). Called by scripts/render_deliverable.py with
+scene=1; the single-body renders are untouched."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physmorph import gpu                                      # noqa: E402,F401  (before torch)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image                                          # noqa: E402
from physmorph.render.exterior import Lattice, make_field      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

dev = torch.device("cuda")


def options(argv):
    opts = dict(s[1:].split("=", 1) for s in argv[1:] if s.startswith("+"))
    return opts, [s for s in argv if not s.startswith("+")]


def base_discs(x, a, field, h_lat):
    """The base display of one body: its exterior's discs (lattice about its own centre) and the twice-averaged
    normals (render_deliverable.Surface's, without the detail layer)."""
    c = x.mean(0)
    r = float((x - c).norm(dim=1).max())
    lat = Lattice(c, 2.8 * r)
    with torch.no_grad():
        p, g, _, _ = lat.discs(make_field(field, x, a, radius=3.0, offset=0.8, device_field=field == "zbs"), h_lat)
    n = nnf.normalize(g, dim=1)
    d, nbx = knn_self_torch(x, 33)
    reach = float(d[(x - x[nbx[:, 1:]].mean(1)).norm(dim=1) >= .5 * a / .708, 32].median())
    spacing = float(knn_self_torch(p, 2)[0][:, 1].median())
    dd, nn = gpu.KNN(p).query(p, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
    wr = (dd.float() <= reach)[..., None]
    for _ in range(2):
        n = nnf.normalize((n[nn] * wr).sum(1), dim=1, eps=1e-9)
    return p, n


def floor_background(cam, floor_y, discs, extent_xz, shade=0.55, res=768):
    """The studio backdrop with the floor y = floor_y: the floor in the backdrop's lower grey, darkened by a soft
    contact shadow (the discs dropped onto it, weight exp(-height / H), blurred by a Gaussian of H; H a 32nd of the
    scene's horizontal extent)."""
    bg = cam.background.clone()
    o = cam.camera
    d = -cam.to_camera                                         # the pixel rays (unit, world)
    t = (floor_y - o[1]) / d[..., 1].clamp(max=-1e-6)
    hit = (d[..., 1] < -1e-6) & (t > 0)
    q = o + t[..., None] * d
    floor_col = bg.new_tensor((.215, .237, .265)) * 1.08
    pts = torch.cat([p for p, _ in discs])
    H = 0.25 * extent_xz / 8.0
    lo = -extent_xz
    cell = 2 * extent_xz / res
    occ = torch.zeros(res, res, device=dev)
    h = (pts[:, 1] - floor_y).clamp_min(0)
    w = torch.exp(-h / H)
    ij = ((pts[:, [0, 2]] - lo) / cell).long().clamp(0, res - 1)
    occ.index_put_((ij[:, 0], ij[:, 1]), w, accumulate=True)
    k = max(1, int(math.ceil(3 * H / cell)))
    g = torch.exp(-0.5 * (torch.arange(-k, k + 1, device=dev) * cell / H) ** 2)
    g = g / g.sum()
    occ = nnf.conv2d(occ[None, None], g[None, None, :, None], padding=(k, 0))
    occ = nnf.conv2d(occ, g[None, None, None, :], padding=(0, k))[0, 0]
    occ = occ / occ.max().clamp_min(1e-9)
    uv = ((q[..., [0, 2]] - lo) / cell).long().clamp(0, res - 1)
    s = occ[uv[..., 0], uv[..., 1]]
    col = floor_col * (1 - shade * s)[..., None]
    fade = (t / t[hit].quantile(0.95).clamp_min(1e-6)).clamp(0, 1)[..., None] if bool(hit.any()) else 0.0
    col = col * (1 - fade) + bg * fade                        # the floor fades into the backdrop far away
    return torch.where(hit[..., None], col, bg)


def _box(x, floor_y, to_floor):
    lo, hi = x.min(0), x.max(0)
    if floor_y > -1e8 and (to_floor or lo[1] - floor_y < hi[1] - lo[1]):
        lo[1] = floor_y
    return np.concatenate([lo, hi]).astype(np.float64)


def _view(b, margin, dist, aspect):
    lo, hi = b[:3], b[3:]
    r = margin * dist * max(0.5 * (hi[1] - lo[1]), 0.5 * max(hi[0] - lo[0], hi[2] - lo[2]) / aspect)
    return torch.tensor(0.5 * (lo + hi), dtype=torch.float32, device=dev), float(r)


def _basis(az, el):
    """The still camera's direction from its target and its forward, right and up axes (StudioRaster's placement)."""
    a, e = math.radians(az), math.radians(el)
    out = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    fwd = -out
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return out, fwd, right, np.cross(right, fwd)


def _still(b, margin, dist, aspect, az, el):
    """A still camera aimed at the centre of the box b (lo, hi), its distance (StudioRaster: 3.6 r from the centre,
    a 30 degree vertical field) grown until every corner of the box projects inside the frame with the margin: the
    composition centred on the box and the whole box in view, perspective included."""
    lo, hi = b[:3], b[3:]
    c = 0.5 * (lo + hi)
    out, fwd, right, up = _basis(az, el)
    ty = math.tan(math.radians(15.0))
    tx = ty * aspect
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    r = dist * max(0.5 * (hi[1] - lo[1]), 0.5 * max(hi[0] - lo[0], hi[2] - lo[2]) / aspect)
    for _ in range(200):
        q = corners - (c + 3.6 * r * out)
        depth = q @ fwd
        if (depth > 0).all() and (np.abs(q @ right / depth) <= tx / margin).all() \
                and (np.abs(q @ up / depth) <= ty / margin).all():
            break
        r *= 1.03
    return torch.tensor(c, dtype=torch.float32, device=dev), float(r)


def width_fraction(x, center, r, aspect, az, el):
    """The share of the frame's width the particles x cover through the camera (centre, r): a record of the framing."""
    out, fwd, right, _ = _basis(az, el)
    q = np.asarray(x, np.float64) - (np.asarray(center, np.float64) + 3.6 * r * out)
    u = (q @ right) / (q @ fwd)
    return float((u.max() - u.min()) / (2.0 * math.tan(math.radians(15.0)) * aspect))


def framing(frames, keep, raw, floor_y, sigma_steps, dist, aspect, mode="static", az=0.0, el=12.0):
    """Each kept frame's (centre, radius) for StudioRaster, whose visible half-height at the centre is about the radius
    (distance 3.6 r, 30 degree field). static (the deliverable's main shot): one camera for the whole run, framing
    both bodies over every frame and the floor they land on, a modest margin, the horizon fixed. contact: one camera
    framing the bodies alone (no floor) over the rendered frames (+frames: a still close-up of that span). follow:
    the box around both bodies at each kept frame (down to the floor once the gap below them is under the box's
    height), its corners smoothed over the kept frames by a Gaussian in simulated steps (a camera that zooms with the
    pair)."""
    if mode == "static":
        b = np.stack([_box(np.asarray(frames[k], np.float32), floor_y, True) for k in range(len(frames))])
        v = _still(np.concatenate([b[:, :3].min(0), b[:, 3:].max(0)]), 1.08, dist, aspect, az, el)
        return [v] * len(keep)
    if mode == "contact":
        b = np.stack([_box(np.asarray(frames[k], np.float32), -1e9, False) for k in keep])     # the bodies only
        v = _still(np.concatenate([b[:, :3].min(0), b[:, 3:].max(0)]), 1.12, dist, aspect, az, el)
        return [v] * len(keep)
    box = np.stack([_box(np.asarray(frames[k], np.float32), floor_y, False) for k in keep])
    t = np.asarray([raw(k) for k in keep], np.float64)
    w = np.exp(-0.5 * ((t[:, None] - t[None]) / max(sigma_steps, 1e-9)) ** 2)
    return [_view(b, 1.25, dist, aspect) for b in (w @ box) / w.sum(1, keepdims=True)]


def main():
    opts, argv = options(sys.argv)
    z = np.load(argv[1])
    out = Path(argv[2])
    (out / "frames").mkdir(parents=True, exist_ok=True)
    field = opts.get("field", "zbs")
    AZ, EL, DIST = float(opts.get("az", 0.)), float(opts.get("el", 12.)), float(opts.get("dist", 1.))
    STRIDE = int(opts.get("stride", 1))
    FLOOR = opts.get("floor", "1") == "1"
    SMOOTH = float(opts.get("smooth", 0.75)) * 240.0           # seconds, in simulated steps (dt 1/240)
    frames, rows = z["frames"], z["rows"]
    floor_y = float(z["floor_y"]) if "floor_y" in z.files else -1e9
    tgts = [torch.as_tensor(np.asarray(z[f"tgt{b}"], np.float32), device=dev) for b in range(len(rows))]
    a = .708 * float(knn_self_torch(tgts[0], 9)[0][:, 8].median())    # the field's pitch: the volume sample's
    h_lat = 0.92 * a
    allx = torch.as_tensor(np.asarray(frames[::max(1, len(frames) // 50)], np.float32), device=dev).reshape(-1, 3)
    lo, hi = allx.min(0).values, allx.max(0).values
    extent_xz = float((hi - lo)[[0, 2]].max())
    raw = (lambda k: int(z["raws"][k])) if "raws" in z.files else (lambda k: k)   # noqa: E731  (the simulated step)
    keep = list(range(0, len(frames), STRIDE))
    if "frames" in opts:
        f0, f1 = (int(v) for v in opts["frames"].split("-"))
        keep = [k for k in keep if f0 <= k <= f1]
    HGT = int(opts.get("res", 2160))                           # the picture's height (16:9); 2160 the deliverable's 4K
    WID = int(round(HGT * 16 / 9))
    views = framing(frames, keep, raw, floor_y, SMOOTH, DIST, WID / HGT, mode=opts.get("camera", "static"),
                    az=AZ, el=EL)
    log = open(out / "rows.txt", "w")
    for k, (center, radius) in zip(keep, views):
        cam = StudioRaster(center, radius, WID, HGT, AZ, EL)
        cam.camera = center + 3.6 * radius * center.new_tensor((math.cos(math.radians(EL)) * math.sin(math.radians(AZ)),
                                                               math.sin(math.radians(EL)),
                                                               math.cos(math.radians(EL)) * math.cos(math.radians(AZ))))
        x = torch.as_tensor(np.asarray(frames[k], np.float32), device=dev)
        discs = [base_discs(x[s:e].contiguous(), a, field, h_lat) for s, e in rows]
        q = torch.cat([p for p, _ in discs])
        n = torch.cat([nn for _, nn in discs])
        if FLOOR and floor_y > -1e8:
            cam.background = floor_background(cam, floor_y, discs, extent_xz)
        rr = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
        tt = nnf.normalize(torch.linalg.cross(n, rr), dim=1, eps=1e-9)
        rot = torch.stack((tt, torch.linalg.cross(n, tt), n), dim=2)
        sig = torch.full((len(q),), h_lat, device=dev)
        cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
        with torch.no_grad():
            img = cam(q, n, cov, torch.full((len(q),), .92, device=dev), normal_kernel=normal_filter_size(HGT, False))
        Image.fromarray((img.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(out / "frames" / f"{raw(k):04d}.jpg",
                                                                                     quality=93)
        wf = width_fraction(x.cpu().numpy(), center.cpu().numpy(), radius, WID / HGT, AZ, EL)
        log.write(json.dumps({"frame": raw(k), "discs": [len(p) for p, _ in discs], "width_frac": round(wf, 4)}) + "\n")
        log.flush()
    print("wrote", out, len(keep), "frames", flush=True)


if __name__ == "__main__":
    main()
