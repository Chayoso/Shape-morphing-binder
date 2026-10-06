"""child_detail_morph.py FRAMES_NPZ DENSE_NPZ[,DENSE_NPZ...] OUT_DIR [+fit=1] [+steps=15] [+stride=1] [+video=1] [+carry=v|h] [+fit_until=1] [+decay=0] — D111 stage 1:
the display surface finer than the body (child_detail_static.py) carried through a finished morph, frame by frame.

The surface carries nothing back to the body (massless, display only), so running it over a run's kept frames is the same
as running it inside the run. Per kept frame (every STRIDE-th and the last), in order:
  base     the exterior's discs of the frame's particles (their Zhu–Bridson field at the run's pitch a) on a lattice of
           0.92 of the dense samples' pitch a_c, with the base display's normals n (the field's, averaged twice over
           its reach: the body's own surface flickers no more than the display's); the slope of h and the footprint S
           read only same-side neighbours (normals within 60 degrees: two parts in contact are not mixed);
  carried  the previous frame's discs moved by the particles' displacement since then (each disc by the Gaussian-weighted
           mean, sigma a, of its 8 nearest particles' moves); each base disc takes v, its Adam moments and step count
           as the tent-weighted (one pixel) mean of the carried discs' around it, 0 where none is within a pixel;
  fit      (fit=1, the render arm) STEPS steps of Adam (lr 0.02 a_c, the moments carried with the discs, so a disc
           already fitted is not restarted at a full step) on v, h = S v with S the render's splat footprint (a tent
           of one pixel), to the render term (silhouette + shading of the drawn discs p + h n, normals tilted by the
           slope of h) against the mean pictures of the dense samples' own exteriors at a pixel of a_c from the
           pipeline's 18 views; the gradient only where the disc stands within one pitch a of the dense sample's
           exterior (the body has arrived there: D105's reference rule), |v| <= a;
  draw     (video=1) the drawn discs at 3840 x 2160 in the studio's look (morph_4k.py's camera), frames/NNNN.jpg.
fit=0 (the physics-only twin): h stays 0, the base discs drawn the same way. Per frame a JSON line in OUT_DIR/rows.txt:
the discs, the arrived share, h's rms and the render terms before and after the frame's fit."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image                                          # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

dev = torch.device("cuda")
opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv if not s.startswith("+")]
CARRY_H = opts.get("carry", "v") == "h"
FIT_UNTIL = float(opts.get("fit_until", 1.0))                # diagnostic: fit only over this share of the frames
DECAY = opts.get("decay", "0") == "1"                        # the step of a disc over 1 / sqrt(its fits so far, in frames)
FIT, STEPS, STRIDE, VIDEO = (opts.get("fit", "1") == "1", int(opts.get("steps", 15)), int(opts.get("stride", 1)),
                             opts.get("video", "1") == "1")
z = np.load(argv[1], allow_pickle=True)
denses = [torch.as_tensor(np.asarray(np.load(p)["tgt"], np.float32), device=dev) for p in argv[2].split(",")]
out = Path(argv[3])
(out / "frames").mkdir(parents=True, exist_ok=True)
cfg = PipelineConfig()
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
pitch = lambda s: .708 * float(knn_self_torch(s, 9)[0][:, 8].median())   # noqa: E731
a, a_c = pitch(tgt), pitch(denses[0])
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
extent = float(tgt.abs().max()) * 1.25
res = int(np.ceil(2.0 * extent / a_c))
pix = 2.0 * extent / res
h_lat = 0.92 * a_c
lat = Lattice(center, 2.8 * radius)
views = make_views(cfg.render_views, cfg.render_elevs)


def exterior(s, a_field):
    with torch.no_grad():
        p, g, _, _ = lat.discs(ZhuBridson(s, a_field), h_lat)
    return p, nnf.normalize(g, dim=1)


per = [exterior_targets(*exterior(s, a_c), views, res, extent, cfg.sil_k, cfg.pbr_ambient) for s in denses]
sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
p_t, _ = exterior(denses[0], a_c)
tree_t = gpu.KNN(p_t)
cam = StudioRaster(center, radius, 3840, 2160, 35., 18.)
print(f"pitch {a:.4f}, dense {a_c:.4f}; render {res} px; fit {FIT} steps {STEPS} stride {STRIDE}", flush=True)
K = 16


class Surface:
    """A frame's base discs, their neighbourhoods, the slope fit and the splat footprint."""

    def __init__(self, x):
        self.x = x
        self.p, n = exterior(x, a)
        M = len(self.p)
        # the base normal: the base display's treatment (the mean over the discs within the reach of the particles' 32
        # nearest, twice), so the body's own surface flickers no more than the display's; the relief's normal is h's
        d, nbx = knn_self_torch(x, 33)
        reach = float(d[(x - x[nbx[:, 1:]].mean(1)).norm(dim=1) >= .5 * a / .708, 32].median())
        spacing = float(knn_self_torch(self.p, 2)[0][:, 1].median())
        dd, nn = gpu.KNN(self.p).query(self.p, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
        wr = (dd.float() <= reach)[..., None]
        for _ in range(2):
            n = nnf.normalize((n[nn] * wr).sum(1), dim=1, eps=1e-9)
        self.n = n
        _, nb = knn_self_torch(self.p, K + 1)
        self.nb = nb[:, 1:]
        side = ((self.n[self.nb] * self.n[:, None]).sum(-1) > 0.5).float()   # same-side neighbours only (a thin part's
        ref = torch.where(self.n[:, :1].abs() < .9, self.n.new_tensor((1., 0., 0.)), self.n.new_tensor((0., 1., 0.))).expand_as(self.n)
        self.t1 = nnf.normalize(torch.linalg.cross(self.n, ref), dim=1)    # two faces, or two parts in contact, not mixed)
        self.t2 = torch.linalg.cross(self.n, self.t1)
        d = self.p[self.nb] - self.p[:, None]
        D = torch.stack(((d * self.t1[:, None]).sum(-1), (d * self.t2[:, None]).sum(-1)), -1)
        w = (torch.exp(-(d.norm(dim=-1) / (1.5 * h_lat)) ** 2) * side)[..., None]
        self.A = torch.linalg.solve((D * w).transpose(1, 2) @ D + 1e-6 * h_lat ** 2 * torch.eye(2, device=dev), (D * w).transpose(1, 2))
        dist = torch.cat((torch.zeros(M, 1, device=dev), d.norm(dim=-1)), 1)
        self.idx = torch.cat((torch.arange(M, device=dev)[:, None], self.nb), 1)
        ws = (1.0 - dist / pix).clamp_min(0.0) * torch.cat((torch.ones(M, 1, device=dev), side), 1)
        self.ws = ws / ws.sum(1, keepdim=True)
        self.arrived = (tree_t.query(self.p, 1)[0].reshape(-1).float() < a)

    def smooth(self, v):
        return (self.ws * v[self.idx]).sum(1)

    def drawn(self, h):
        g2 = (self.A @ (h[self.nb] - h[:, None])[..., None]).squeeze(-1)
        n = nnf.normalize(self.n - g2[:, :1] * self.t1 - g2[:, 1:] * self.t2, dim=1)
        return self.p + h[:, None] * self.n, n


def carried(prev, prev_state, cur):
    """The per-disc state (v and its Adam moments and step count, (M, 4)) on cur's discs: prev's discs moved by the
    particles' displacement, read by the one-pixel tent (v, not h = S v, is carried, so that the footprint is applied
    once, not once a frame; the moments too, so that a disc already fitted is not restarted at a full step)."""
    disp = cur.x - prev.x                                      # the same particles, in order
    dd, ii = gpu.KNN(prev.x).query(prev.p, 8)
    w = torch.exp(-(dd.float() / a) ** 2)
    moved = prev.p + (w[..., None] * disp[ii]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
    dd, ii = gpu.KNN(moved).query(cur.p, 4)
    w = (1.0 - dd.float() / pix).clamp_min(0.0)
    s = w.sum(1, keepdim=True)
    return torch.where(s > 0, (w[..., None] * prev_state[ii]).sum(1) / s.clamp_min(1e-12),
                       torch.zeros(len(cur.p), prev_state.shape[1], device=dev))


def draw(q, n, path):
    rr = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    tt = nnf.normalize(torch.linalg.cross(n, rr), dim=1, eps=1e-9)
    rot = torch.stack((tt, torch.linalg.cross(n, tt), n), dim=2)
    sig = torch.full((len(q),), h_lat, device=dev)
    cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
    with torch.no_grad():
        img = cam(q, n, cov, torch.full((len(q),), .92, device=dev), normal_kernel=normal_filter_size(2160, False))
    Image.fromarray((img.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(path, quality=93)


n_frames = len(z["raws"])
keep = list(range(0, n_frames, STRIDE)) + ([n_frames - 1] if (n_frames - 1) % STRIDE else [])
prev = prev_state = None
rows = open(out / "rows.txt", "w")
B1, B2, LR = 0.9, 0.999, 0.02 * a_c
for i, k in enumerate(keep):
    cur = Surface(torch.as_tensor(np.asarray(z["frames"][k], np.float32), device=dev))
    state = torch.zeros(len(cur.p), 4, device=dev) if prev is None else carried(prev, prev_state, cur)
    v = state[:, 0].clone().requires_grad_(True)
    m1, m2, cnt = state[:, 1].clone(), state[:, 2].clone(), state[:, 3].clone()
    with torch.no_grad():
        before = [float(t) for t in d_exterior(*cur.drawn(cur.smooth(v)), sils, shade, views, res, extent, cfg.sil_k,
                                                ambient=cfg.pbr_ambient)]
    if FIT and STEPS > 0 and bool(cur.arrived.any()) and i <= FIT_UNTIL * (len(keep) - 1):
        live = cur.arrived.float()                             # only where the body stands on the target
        for _ in range(STEPS):
            ls, lp = d_exterior(*cur.drawn(cur.smooth(v)), sils, shade, views, res, extent, cfg.sil_k, ambient=cfg.pbr_ambient)
            g, = torch.autograd.grad(ls + cfg.w_pbr * lp, v)
            with torch.no_grad():                              # Adam with the moments carried by the discs
                g = g * live
                m1 = torch.where(live > 0, B1 * m1 + (1 - B1) * g, m1)
                m2 = torch.where(live > 0, B2 * m2 + (1 - B2) * g * g, m2)
                cnt = cnt + live
                c = cnt.clamp_min(1.0)
                lr = LR / (c / STEPS).clamp_min(1.0).sqrt() if DECAY else LR   # +decay=1: a disc fitted over many frames
                step = lr * (m1 / (1 - B1 ** c)) / ((m2 / (1 - B2 ** c)).sqrt() + 1e-12)   # takes ever smaller steps
                v -= step * live
                v.clamp_(-a, a)
    with torch.no_grad():
        h = cur.smooth(v)
        q, n = cur.drawn(h)
        after = [float(t) for t in d_exterior(q, n, sils, shade, views, res, extent, cfg.sil_k, ambient=cfg.pbr_ambient)]
        if VIDEO:
            draw(q, n, out / "frames" / f"{i:04d}.jpg")
    row = dict(frame=i, raw=int(z["raws"][k]), discs=len(cur.p), arrived=float(cur.arrived.float().mean()),
               h_rms=float(h.square().mean().sqrt()) / a, sil_before=before[0], pbr_before=before[1], sil_after=after[0],
               pbr_after=after[1])
    rows.write(json.dumps(row) + "\n")
    rows.flush()
    if i % 20 == 0 or k == n_frames - 1:
        print(json.dumps(row), flush=True)
    carry = h if CARRY_H else v.detach()                     # +carry=h: the drawn height (smooth at the pixel) is carried
    prev, prev_state = cur, torch.stack((carry, m1, m2, cnt), 1)
torch.save(dict(state=prev_state.cpu(), p=prev.p.cpu(), n=prev.n.cpu()), out / "last.pt")
print("wrote", out, len(keep), "frames", flush=True)
