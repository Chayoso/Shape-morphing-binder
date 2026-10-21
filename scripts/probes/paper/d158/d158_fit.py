"""d158_fit.py FRAMES12_NPZ DENSE_NPZ[,DENSE_NPZ...] OUT_DIR +mesh=bunny +seed=97 [+dump=10,20,40,80,-1] [+res_loop=188]
[+draws=8] [+sfit=80,-1] [+sfit_steps=200] [+fit=1 +steps=15 +carry=h +decay=1 +video=0] -- D158, a measurement only.

A scratch copy of the tag's scripts/render_deliverable.py (repo_freeze1015, the deliverable's renderer) with D115's fit
unchanged (+fit=1 +steps=15 +carry=h +decay=1: base discs of the frame's particles at the run's pitch a on a lattice of
0.92 a_c, v carried with the discs, 15 Adam steps a frame on the render term against the dense samples' pictures at a
pixel of a_c, the gradient only where the body has arrived, h = S v). Nothing is drawn (+video=0). At the kept frames in
+dump (kept-frame indices, 20 a second; -1 = the last) the frame's layer height h is taken apart:

  h_uW   what an outer-layer per-particle normal displacement u at the run's pitch can represent, through the display's
         own particle -> disc weighting (carried(): each disc by the Gaussian-weighted mean, sigma a, of its 8 nearest
         particles' moves; a particle off the outer layer does not move): least squares of h over u on the layer
         particles (the pipeline's layer, window/layer.layer_relax_data on the frame), u along the layer's normals;
  h_uF   the same through the exterior field's own response (the path the loop's render gradient takes to u: a disc
         moves along its normal by -(|p - xbar| - offset) / slope, Tracked.read, linearised in the particles, Zhu and
         Bridson's kernel of 3 pitches);
  rest   h - h_uW and h - h_uF;
  bands  h, each part and the render gradient split by Gaussian means over the discs (same-side neighbours) of sigma
         0.5 / 1 / 2 / 4 pitches: wavelengths below about 2.7 / 2.7-5.4 / 5.4-10.8 / 10.8-21.6 / above 21.6 pitches (the
         family of D107); h_low = the part above 10.8 pitches (the exterior's offset and the shape), h_rel = h - h_low.
Each part is drawn as the fit draws h (p + part n, normals tilted by its slope) and its render terms read at
  dense_ac    the dense samples' pictures at a pixel of a_c (D115's own targets),
  dense_loop  the same pictures at the loop's resolution (+res_loop px over the same extent: the run's loss pixel),
  loop_ac / loop_loop  the run's own loss targets (its target sample and further draws, seeds seed + 1 + 100 k, at the
              run's pitch a, as prepare.py / target.py draw them) at either pixel, on the same lattice as the state's
              discs (like for like; the loop's own lattice is half its pixel).
Also at each setting the render gradient at h = 0 (where the body has arrived): its cosine with h and with h_rel and its
band shares; and at the frames in +sfit a static fit from h = 0 (sfit_steps Adam steps, the fit's own rule) at each
setting: what each picture asks of a free layer. One JSON line per dump in OUT_DIR/d158.jsonl; the arrays in
OUT_DIR/dump_NNNN.npz; the fit's per-frame rows in OUT_DIR/rows.txt as render_deliverable writes them."""
import json, os, sys, time
from pathlib import Path
REPO = os.environ.get("D158_REPO", "/data/relcfd/chayo/physmorph_v2/repo_freeze1015")
sys.path.insert(0, REPO)                                     # the tag's code
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing  # noqa: E402
from physmorph.render.exterior import Lattice, make_field      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, load_normalized  # noqa: E402

dev = torch.device("cuda")
opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv if not s.startswith("+")]
CARRY_H = opts.get("carry", "v") == "h"
FIT_UNTIL = float(opts.get("fit_until", 1.0))
DECAY = opts.get("decay", "0") == "1"
FIT, STEPS, STRIDE = opts.get("fit", "1") == "1", int(opts.get("steps", 15)), int(opts.get("stride", 1))
RAD, OFF = float(opts.get("radius", 3.)), float(opts.get("offset", .8))
FIELD = opts.get("field", "zb")
FRAMES = [int(v) for v in opts["frames"].split("-")] if "frames" in opts else None
MESH, SEED = opts["mesh"], int(opts.get("seed", 97))
RES_LOOP, NDRAW = int(opts.get("res_loop", 188)), int(opts.get("draws", 8))
DUMP = [int(v) for v in opts.get("dump", "10,20,40,80,-1").split(",")]
SFIT = [int(v) for v in opts.get("sfit", "80,-1").split(",") if v]
SFIT_STEPS = int(opts.get("sfit_steps", 200))
t_start = time.time()
z = np.load(argv[1], allow_pickle=True)
denses = [torch.as_tensor(np.asarray(np.load(p)["tgt"], np.float32), device=dev) for p in argv[2].split(",")]
out = Path(argv[3])
out.mkdir(parents=True, exist_ok=True)
cfg = PipelineConfig()
tgt_np = np.asarray(z["tgt"], np.float32)
tgt = torch.as_tensor(tgt_np, device=dev)
FR, RAWS = z["frames"], np.asarray(z["raws"])                 # read once (each z[...] reads the whole array)
X_END = torch.as_tensor(np.asarray(FR[-1], np.float32), device=dev)
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
        p, g, _, _ = lat.discs(make_field(FIELD, s, a_field, radius=RAD, offset=OFF, device_field=FIELD == "zbs"), h_lat)
    return p, nnf.normalize(g, dim=1)


def mean_pics(exts, R):
    per = [exterior_targets(pp, nn, views, R, extent, cfg.sil_k, cfg.pbr_ambient) for pp, nn in exts]
    return ([torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])],
            [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])])


dense_ext = [exterior(s, a_c) for s in denses]
sils, shade = mean_pics(dense_ext, res)                       # D115's targets (the fit's)
p_t = dense_ext[0][0]
tree_t = gpu.KNN(p_t)
# the run's own loss targets: its target sample and further draws, as prepare.py draws them (d157_bias.py's rebuild)
os.chdir(REPO)
n_run = len(tgt_np)
fill_src, fill_tgt, frame = {}, {}, {}
_, v_src = load_normalized("assets/isosphere.obj", n_run, SEED, return_volume=True, sample="stratified", fill=fill_src)
tg, _ = load_normalized(f"assets/{MESH}.obj", n_run, SEED + 1, match_volume=v_src, sample="stratified",
                        return_volume=True, frame=frame, fill=fill_tgt)
tgt_same = float(np.abs(tg - tgt_np).max())
draws = draws_in_frame(f"assets/{MESH}.obj", frame, n_run, [SEED + 1 + 100 * k for k in range(1, NDRAW)]) if NDRAW > 1 else []
loop_ext = [exterior(s, a) for s in [tgt] + [torch.as_tensor(np.asarray(d, np.float32), device=dev) for d in draws]]
SETTINGS = {"dense_ac": (sils, shade, res), "dense_loop": (*mean_pics(dense_ext, RES_LOOP), RES_LOOP),
            "loop_ac": (*mean_pics(loop_ext, res), res), "loop_loop": (*mean_pics(loop_ext, RES_LOOP), RES_LOOP)}
del loop_ext
tdiff = {}
for nm in ("ac", "loop"):                                     # how far the loop's pictures are from the dense ones
    A, B = SETTINGS["dense_" + nm], SETTINGS["loop_" + nm]
    tdiff[nm] = dict(sil=float(torch.stack([(x - y).square().mean() for x, y in zip(A[0], B[0])]).mean()),
                     pbr=float(torch.stack([(x - y).square().mean() for x, y in zip(A[1], B[1])]).mean()))
head = dict(mesh=MESH, seed=SEED, pitch_a=a, pitch_ac=a_c, ac_over_a=a_c / a, extent=extent, res_ac=res,
            pix_ac=pix, pix_ac_over_a=pix / a, res_loop=RES_LOOP, pix_loop=2 * extent / RES_LOOP,
            pix_loop_over_a=2 * extent / RES_LOOP / a, lattice_over_a=h_lat / a,
            loop_lattice_over_a=min(extent / RES_LOOP, 0.92 * a) / a, draws=1 + len(draws), tgt_reproduced_maxdiff=tgt_same,
            target_pictures_diff=tdiff, dense=argv[2].split(","), frames12=argv[1])
print("D158 head", json.dumps(head), flush=True)
with open(out / "head.json", "w") as f:
    json.dump(head, f, indent=1)
K = 16


class Surface:
    """A frame's base discs, their neighbourhoods, the slope fit and the splat footprint (render_deliverable's)."""

    def __init__(self, x):
        self.x = x
        self.p, n = exterior(x, a)
        M = len(self.p)
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
        side = ((self.n[self.nb] * self.n[:, None]).sum(-1) > 0.5).float()
        self.side = side
        ref = torch.where(self.n[:, :1].abs() < .9, self.n.new_tensor((1., 0., 0.)), self.n.new_tensor((0., 1., 0.))).expand_as(self.n)
        self.t1 = nnf.normalize(torch.linalg.cross(self.n, ref), dim=1)
        self.t2 = torch.linalg.cross(self.n, self.t1)
        d = self.p[self.nb] - self.p[:, None]
        D = torch.stack(((d * self.t1[:, None]).sum(-1), (d * self.t2[:, None]).sum(-1)), -1)
        w = (torch.exp(-(d.norm(dim=-1) / (1.5 * h_lat)) ** 2) * side)[..., None]
        self.A = torch.linalg.solve((D * w).transpose(1, 2) @ D + 1e-6 * h_lat ** 2 * torch.eye(2, device=dev), (D * w).transpose(1, 2))
        dist = torch.cat((torch.zeros(M, 1, device=dev), d.norm(dim=-1)), 1)
        self.dist = dist
        self.idx = torch.cat((torch.arange(M, device=dev)[:, None], self.nb), 1)
        ws = (1.0 - dist / pix).clamp_min(0.0) * torch.cat((torch.ones(M, 1, device=dev), side), 1)
        self.ws = ws / ws.sum(1, keepdim=True)
        dd, ii = gpu.KNN(x).query(self.p, 8)
        w = torch.exp(-(dd.float() / a) ** 2)
        left = (w * (x - X_END).norm(dim=1)[ii]).sum(1) / w.sum(1).clamp_min(1e-12)
        self.arrived = (tree_t.query(self.p, 1)[0].reshape(-1).float() < a) & (left < a)

    def smooth(self, v):
        return (self.ws * v[self.idx]).sum(1)

    def drawn(self, h):
        g2 = (self.A @ (h[self.nb] - h[:, None])[..., None]).squeeze(-1)
        n = nnf.normalize(self.n - g2[:, :1] * self.t1 - g2[:, 1:] * self.t2, dim=1)
        return self.p + h[:, None] * self.n, n


def carried(prev, prev_state, cur):
    disp = cur.x - prev.x
    dd, ii = gpu.KNN(prev.x).query(prev.p, 8)
    w = torch.exp(-(dd.float() / a) ** 2)
    moved = prev.p + (w[..., None] * disp[ii]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
    dd, ii = gpu.KNN(moved).query(cur.p, 4)
    w = (1.0 - dd.float() / pix).clamp_min(0.0)
    s = w.sum(1, keepdim=True)
    return torch.where(s > 0, (w[..., None] * prev_state[ii]).sum(1) / s.clamp_min(1e-12),
                       torch.zeros(len(cur.p), prev_state.shape[1], device=dev))


# ---------------------------------------------------------------- D158's readings
SIGMAS = (0.5, 1.0, 2.0, 4.0)                                 # pitches; band edges ~ 5.4 sigma in wavelength
BANDS = ("lt2.7", "2.7-5.4", "5.4-10.8", "10.8-21.6", "gt21.6")


class Diffusion:
    """Gaussian means over the discs: repeated weighted means over each disc and its same-side 16 nearest, the number
    of passes from the measured variance of one pass (per tangent axis)."""

    def __init__(self, cur):
        w = torch.cat((torch.ones(len(cur.p), 1, device=dev),
                       torch.exp(-0.5 * (cur.dist[:, 1:] / h_lat) ** 2) * cur.side), 1)
        self.w = w / w.sum(1, keepdim=True)
        self.idx = cur.idx
        self.var = float((self.w * cur.dist.square()).sum(1).median()) / 2.0
        self.steps = [max(1, int(round((s * a) ** 2 / self.var))) for s in SIGMAS]

    def means(self, h):
        out, g, done = [], h, 0
        for n in self.steps:
            for _ in range(n - done):
                g = (self.w * g[self.idx]).sum(1)
            done = n
            out.append(g)
        return out                                             # G_0.5, G_1, G_2, G_4

    def bands(self, h):
        g = self.means(h)
        return [h - g[0], g[0] - g[1], g[1] - g[2], g[2] - g[3], g[3]], g[2]


def cg_lsq(A, At, b, ncol, iters=500, tol=1e-6):
    """min |A u - b|^2 + eps |u|^2 by conjugate gradients on the normal equations (eps 1e-3 of the median column
    weight of the columns that carry any): the projection of b on what A can represent."""
    colw = At(torch.ones_like(b), sq=True)
    live = colw > 0
    eps = 1e-3 * float(colw[live].median()) if bool(live.any()) else 1.0
    rhs = At(b)
    u = torch.zeros(ncol, device=dev, dtype=b.dtype)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum()
    r0 = float(rs.sqrt())
    it = 0
    for it in range(iters):
        Ap = At(A(p)) + eps * p
        al = rs / (p * Ap).sum().clamp_min(1e-30)
        u += al * p
        r -= al * Ap
        rn = (r * r).sum()
        if float(rn.sqrt()) < tol * max(r0, 1e-30):
            break
        p = r + (rn / rs) * p
        rs = rn
    return u, A(u), it + 1, int(live.sum())


class Map:
    """A sparse disc <- particle map (vals, idx: (M, K)), A u and A^T r."""

    def __init__(self, vals, idx, N):
        self.v, self.i, self.N = vals.double(), idx.clamp_min(0), N

    def A(self, u):
        return (self.v * u[self.i]).sum(1)

    def At(self, r, sq=False):
        val = (self.v.square() if sq else self.v * r[:, None]).reshape(-1)
        return torch.zeros(self.N, device=dev, dtype=torch.float64).index_add_(0, self.i.reshape(-1), val)


def maps(cur):
    """The two particle -> disc maps of an outer-layer normal displacement u (per particle, along its layer normal)."""
    x, p, n = cur.x, cur.p, cur.n
    N = len(x)
    sp = layer_spacing(x)
    lmask, lnrm, _, _ = layer_relax_data(x, sp, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
    # (W) the display's own weighting: each disc moves by the Gaussian-weighted mean (sigma a) of its 8 nearest moves
    dd, ii = gpu.KNN(x).query(p, 8)
    w = torch.exp(-(dd.float() / a) ** 2)
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    Wv = w * (lnrm[ii] * n[:, None]).sum(-1) * lmask[ii]
    # (F) the exterior field's linear response (Tracked.read at the disc, Zhu and Bridson's kernel): d move_i / d x_k
    field = make_field(FIELD, x, a, radius=RAD, offset=OFF)
    fi = field.neighbours(p, 0.0)
    R = field.radius
    _, g, _ = field(p)
    slope = g.norm(dim=1).clamp_min(1e-6)
    n0 = g / slope[:, None]
    Fv = torch.zeros(fi.shape, device=dev)
    for s in range(0, len(p), 20000):
        sl = slice(s, s + 20000)
        idx = fi[sl]
        valid = (idx >= 0).float()
        P = x[idx.clamp_min(0)]
        uu = p[sl, None] - P
        t = (1. - uu.square().sum(-1) / (R * R)).clamp_min(0.) * valid
        wk = t ** 3
        sw = wk.sum(1).clamp_min(1e-12)
        xbar = (wk[..., None] * P).sum(1) / sw[:, None]
        dvec = p[sl] - xbar
        dhat = dvec / dvec.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gw = (6. / (R * R)) * (t * t)[..., None] * uu                      # grad of w_k in x_k
        J = (wk[..., None] * dhat[:, None] + (dhat[:, None] * (P - xbar[:, None])).sum(-1, keepdim=True) * gw) \
            / (sw * slope[sl])[:, None, None]
        Fv[sl] = (J * lnrm[idx.clamp_min(0)]).sum(-1) * lmask[idx.clamp_min(0)] * valid \
            * (n0[sl] * n[sl]).sum(1, keepdim=True)
    chk = None
    if not getattr(maps, "checked", False):                    # the linearisation against autograd, once
        sub = torch.arange(0, len(p), max(1, len(p) // 4000), device=dev)
        xx = x.clone().requires_grad_(True)
        idx = fi[sub]
        valid = (idx >= 0).float()
        P = xx[idx.clamp_min(0)]
        uu = p[sub, None] - P
        t = (1. - uu.square().sum(-1) / (R * R)).clamp_min(0.) * valid
        wk = t ** 3
        xbar = (wk[..., None] * P).sum(1) / wk.sum(1).clamp_min(1e-12)[:, None]
        move = -((p[sub] - xbar).norm(dim=1) - field.offset) / slope[sub] * (n0[sub] * n[sub]).sum(1)
        rr = torch.randn(len(sub), device=dev)
        G, = torch.autograd.grad((move * rr).sum(), xx)
        ga = (G * lnrm).sum(1) * lmask
        Fs = Map(Fv[sub], fi[sub], N)
        an = Fs.At(rr.double()).float()
        chk = float((an - ga).norm() / ga.norm().clamp_min(1e-30))
        maps.checked = True
    return (Map(Wv, ii, N), Map(Fv, fi, N), lmask, lnrm, sp, chk, int((fi >= 0).sum(1).float().mean()))


def terms(cur, hh, st):
    q, nn_ = cur.drawn(hh)
    ls, lp = d_exterior(q, nn_, st[0], st[1], views, st[2], extent, cfg.sil_k, ambient=cfg.pbr_ambient)
    return float(ls), float(lp)


def grad_at_zero(cur, st):
    hh = torch.zeros(len(cur.p), device=dev, requires_grad=True)
    q, nn_ = cur.drawn(hh)
    ls, lp = d_exterior(q, nn_, st[0], st[1], views, st[2], extent, cfg.sil_k, ambient=cfg.pbr_ambient)
    g, = torch.autograd.grad(ls + cfg.w_pbr * lp, hh)
    return -g * cur.arrived.float()


def static_fit(cur, st, steps):
    """From h = 0: the fit's own rule (v, h = S v, Adam lr 0.02 a_c, the gradient where the body has arrived, |v| <= a)."""
    live = cur.arrived.float()
    v = torch.zeros(len(cur.p), device=dev, requires_grad=True)
    m1 = torch.zeros_like(v)
    m2 = torch.zeros_like(v)
    for c in range(1, steps + 1):
        q, nn_ = cur.drawn(cur.smooth(v))
        ls, lp = d_exterior(q, nn_, st[0], st[1], views, st[2], extent, cfg.sil_k, ambient=cfg.pbr_ambient)
        g, = torch.autograd.grad(ls + cfg.w_pbr * lp, v)
        with torch.no_grad():
            g = g * live
            m1 = B1 * m1 + (1 - B1) * g
            m2 = B2 * m2 + (1 - B2) * g * g
            v -= LR * (m1 / (1 - B1 ** c)) / ((m2 / (1 - B2 ** c)).sqrt() + 1e-12) * live
            v.clamp_(-a, a)
    with torch.no_grad():
        return cur.smooth(v)


def analyse(cur, h, k, i, sfit):
    t0 = time.time()
    arr = cur.arrived
    on = arr.float()
    E = lambda t: float((t.double().square() * on).sum() / on.sum().clamp_min(1))      # noqa: E731  energy, arrived
    cos = lambda s, t: float((s * t * on).sum() / ((s * s * on).sum().sqrt() * (t * t * on).sum().sqrt()).clamp_min(1e-30))  # noqa: E731
    dif = Diffusion(cur)
    hb, h_low = dif.bands(h)
    h_rel = h - h_low
    Wm, Fm, lmask, lnrm, sp, chk, fk = maps(cur)
    N = len(cur.x)
    parts, fits = {}, {}
    for nm, mp in (("W", Wm), ("F", Fm)):
        for src, b in (("h", h), ("rel", h_rel)):
            u, hu, its, ncol = cg_lsq(mp.A, mp.At, b.double(), N)
            parts[f"{src}_u{nm}"] = hu.float()
            fits[f"{src}_u{nm}"] = dict(iters=its, cols=ncol, u_rms_sp=float(u[lmask > 0].square().mean().sqrt()) / sp,
                                        u_p99_sp=float(u[lmask > 0].abs().quantile(0.99)) / sp)
            if src == "h":
                fits[f"{src}_u{nm}"]["u"] = u.float()
    hu = {nm: parts[f"h_u{nm}"] for nm in "WF"}
    variants = {"zero": torch.zeros_like(h), "h": h,
                "uW": hu["W"], "restW": h - hu["W"], "uF": hu["F"], "restF": h - hu["F"],
                "low": h_low, "low+relW": h_low + parts["rel_uW"], "low+relF": h_low + parts["rel_uF"], "rel": h_rel}
    row = dict(frame=i, kept=k, t_s=k / 20.0, raw=int(RAWS[k]), discs=len(cur.p), arrived=float(on.mean()),
               layer=int(lmask.sum()), layer_spacing=sp, field_nbrs_mean=fk, lin_check_relerr=chk,
               diffusion_var_over_a2=dif.var / a ** 2, diffusion_steps=dif.steps,
               h_mean_a=float((h * on).sum() / on.sum()) / a, h_rms_a=E(h) ** .5 / a, h_rel_rms_a=E(h_rel) ** .5 / a,
               energy={}, bands={}, terms={}, grad={}, lsq={kk: {q: v for q, v in vv.items() if q != "u"} for kk, vv in fits.items()})
    for nm in "WF":
        for src, b in (("h", h), ("rel", h_rel)):
            pu = parts[f"{src}_u{nm}"]
            row["energy"][f"{src}_u{nm}"] = E(pu) / max(E(b), 1e-30)
            row["energy"][f"{src}_rest{nm}"] = E(b - pu) / max(E(b), 1e-30)
    for nm, t in (("h", h), ("uW", hu["W"]), ("restW", h - hu["W"]), ("uF", hu["F"]), ("restF", h - hu["F"]),
                  ("rel_uW", parts["rel_uW"]), ("rel_uF", parts["rel_uF"])):
        bb, _ = dif.bands(t)
        row["bands"][nm] = [E(x) / a ** 2 for x in bb]           # band energies in pitches^2 (arrived discs)
    for sname, st in SETTINGS.items():
        with torch.no_grad():
            row["terms"][sname] = {vn: terms(cur, vv, st) for vn, vv in variants.items()}
        gneg = grad_at_zero(cur, st)
        gb, _ = dif.bands(gneg)
        tot = sum(E(x) for x in gb)
        row["grad"][sname] = dict(cos_h=cos(gneg, h), cos_rel=cos(gneg, h_rel), cos_uW=cos(gneg, hu["W"]),
                                  band_share=[E(x) / max(tot, 1e-30) for x in gb])
    dump = dict(p=cur.p, n=cur.n, h=h, arrived=arr, x=cur.x, lmask=lmask, lnrm=lnrm, h_low=h_low, h_rel=h_rel,
                h_uW=hu["W"], h_uF=hu["F"], rel_uW=parts["rel_uW"], rel_uF=parts["rel_uF"],
                u_W=fits["h_uW"]["u"], u_F=fits["h_uF"]["u"])
    if sfit:
        row["static_fit"] = {}
        for sname, st in SETTINGS.items():
            hs = static_fit(cur, st, SFIT_STEPS)
            sb, s_low = dif.bands(hs)
            with torch.no_grad():
                tt = terms(cur, hs, st)
                tt_d = terms(cur, hs, SETTINGS["dense_ac"])
            row["static_fit"][sname] = dict(rms_a=E(hs) ** .5 / a, mean_a=float((hs * on).sum() / on.sum()) / a,
                                            bands=[E(x) / a ** 2 for x in sb], corr_h=cos(hs - hs.mean(), h - h.mean()),
                                            corr_rel=cos(hs - s_low, h_rel), terms_own=tt, terms_dense_ac=tt_d)
            dump[f"sfit_{sname}"] = hs
    np.savez_compressed(out / f"dump_{k:04d}.npz", **{kk: vv.detach().float().cpu().numpy() for kk, vv in dump.items()})
    row["analyse_s"] = time.time() - t0
    with open(out / "d158.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("D158 dump", k, json.dumps({q: row[q] for q in ("t_s", "arrived", "h_rms_a", "h_rel_rms_a", "energy", "analyse_s")}),
          flush=True)


n_frames = len(RAWS)
keep = list(range(0, n_frames, STRIDE)) + ([n_frames - 1] if (n_frames - 1) % STRIDE else [])
if FRAMES is not None:
    keep = [k for k in keep if FRAMES[0] <= k <= FRAMES[1]]
DUMP = sorted({(keep[-1] if d < 0 else d) for d in DUMP})
SFIT = {(keep[-1] if d < 0 else d) for d in SFIT}
prev = prev_state = None
rows = open(out / "rows.txt", "w")
B1, B2, LR = 0.9, 0.999, 0.02 * a_c
for i, k in enumerate(keep):
    cur = Surface(torch.as_tensor(np.asarray(FR[k], np.float32), device=dev))
    state = torch.zeros(len(cur.p), 4, device=dev) if prev is None else carried(prev, prev_state, cur)
    v = state[:, 0].clone().requires_grad_(True)
    m1, m2, cnt = state[:, 1].clone(), state[:, 2].clone(), state[:, 3].clone()
    with torch.no_grad():
        before = [float(t) for t in d_exterior(*cur.drawn(cur.smooth(v)), sils, shade, views, res, extent, cfg.sil_k,
                                                ambient=cfg.pbr_ambient)]
    if FIT and STEPS > 0 and bool(cur.arrived.any()) and i <= FIT_UNTIL * (len(keep) - 1):
        live = cur.arrived.float()
        for _ in range(STEPS):
            ls, lp = d_exterior(*cur.drawn(cur.smooth(v)), sils, shade, views, res, extent, cfg.sil_k, ambient=cfg.pbr_ambient)
            g, = torch.autograd.grad(ls + cfg.w_pbr * lp, v)
            with torch.no_grad():
                g = g * live
                m1 = torch.where(live > 0, B1 * m1 + (1 - B1) * g, m1)
                m2 = torch.where(live > 0, B2 * m2 + (1 - B2) * g * g, m2)
                cnt = cnt + live
                c = cnt.clamp_min(1.0)
                lr = LR / (c / STEPS).clamp_min(1.0).sqrt() if DECAY else LR
                step = lr * (m1 / (1 - B1 ** c)) / ((m2 / (1 - B2 ** c)).sqrt() + 1e-12)
                v -= step * live
                v.clamp_(-a, a)
    with torch.no_grad():
        h = cur.smooth(v)
        q, n = cur.drawn(h)
        after = [float(t) for t in d_exterior(q, n, sils, shade, views, res, extent, cfg.sil_k, ambient=cfg.pbr_ambient)]
    row = dict(frame=i, raw=int(RAWS[k]), discs=len(cur.p), arrived=float(cur.arrived.float().mean()),
               h_rms=float(h.square().mean().sqrt()) / a, sil_before=before[0], pbr_before=before[1], sil_after=after[0],
               pbr_after=after[1])
    rows.write(json.dumps(row) + "\n")
    rows.flush()
    if i % 20 == 0 or k == n_frames - 1:
        print(json.dumps(row), f"{time.time() - t_start:.0f} s", flush=True)
    if k in DUMP:
        analyse(cur, h.detach(), k, i, k in SFIT)
    carry = h if CARRY_H else v.detach()
    prev, prev_state = cur, torch.stack((carry, m1, m2, cnt), 1)
print("wrote", out, len(keep), "frames", f"{time.time() - t_start:.0f} s", flush=True)
