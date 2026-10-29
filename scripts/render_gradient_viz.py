"""render_gradient_viz.py VIZ_NPZ DENSE_NPZ[,DENSE_NPZ...] OUT_DIR [+viz=A,B,C,D,U] [+views=35,90] [+res=1080]
[+grid=1] [+interp=1] [+frames=a-b] [+stride=1] [+field=zbs] [+el=18] [+dist=1.05] [+glyph_cell=5] [+arrow_px=46] [+c_avg=1] [+d_avg=1] [+gridfull=1] [+mesh=bunny]
-- the render gradient's videos (RGV): four questions about the render loss and the stress control dF_c, one panel
each, drawn on the deforming body in the deliverable's studio look (scripts/render_deliverable.py's base display,
+fit=0: the exterior's discs of each kept frame's particles, their twice-averaged normals, the same camera, lights,
backdrop and disc footprint), and the u channel. Also reached as render_deliverable.py ... +viz=A (off by default).

  A  "Where does the render act on dF_c?"  the render's share of the pull at each particle,
     s = |lambda g_render| / (|lambda g_render| + |g_phys|): plain grey = physics only, deep orange = render dominates;
  B  "How strong is the render's pull?"  |lambda g_render| on ONE log scale for the whole run (4 decades below the
     run's 99.5th percentile on the outer layer, light lavender = weak, deep violet = strong), and an inset of the
     total render and physics pulls over the morph with a cursor at the current time;
  B2 the same pull against its typical value now: divided by the outer layer's median of it (a running median over
     5 windows, interpolated in time), 1/10 to 10 times, with the same inset (B and B2 are alternatives);
  C  "Where do the render and the physics disagree?"  the cosine at each particle between lambda g_render (before
     PCGrad) and g_phys, as 9-vectors (their means over the steps): blue = the same way, light neutral = unrelated,
     red = against each other; the plain body (hatched) where the render's share is under 5 %; +c_avg=K: the
     cosine's running mean over K windows (it changes much from one window to the next);
  D  "What did the render do to dF_c?"  how much the render changed the particle's control, |Delta dF_c| /
     |dF_c(lambda = 0)| of the 20-step means (the same window optimised twice from the same start), linear 0-100 %,
     plain grey to deep teal; the plain body (hatched) where |dF_c(lambda = 0)| is under 1 % of its 99th percentile
     on the outer layer; 2D arrows on a material grid (glyph_cell pitches on the source positions, drawn where at the
     surface, facing the camera and not hidden, the change over 20 %; arrow_px long on the screen at 1080 lines,
     shortened by the foreshortening, each fading in and out with its change (15-25 %) and its foreshortening, at
     most one per 1.15 lengths in a fixed order of the particles): the principal axis of
     sym(Delta dF_c) along the surface, white double-headed = stretch added, dark converging = compression added;
     +d_avg=K: Delta dF_c and |dF_c(lambda = 0)| as running means over K windows (colour and arrows);
  U  g Delta u = g (u(render on) - u(lambda = 0)) in pitches, blue inward, red outward, one linear scale for the run;
     the plain body (hatched) where u does not act (outside the outer layer or its gate closed).
g = dL/d dF_c of the window's first gradient summed over its 20 driven steps (the render's after PCGrad and times
lambda where named so). +grid=1 (with A-D) also writes the four panels as one 2 x 2 frame (B2 in B's place when B is
not asked; 2W x 2H, each panel at its
own resolution; +gridfull=0 skips it, for 4K panels) and its half-size copy.

VIZ_NPZ: scripts/probes/paper/rgv_values.py (the measurement copy repo_rgv's records per committed window). Time:
+interp=1 (default) interpolates every particle's value linearly between window centres (a window is 20 driven + 20
released steps, 1/6 s); +interp=0 holds each window's value through its frames (the raw version). No colour reference
is rescaled per window. The colour is display only: each disc takes the Gaussian-weighted (sigma = the run's pitch)
mean of its 8 nearest particles' values; StudioRaster draws the discs with that albedo (its albedo=None path, the
deliverable's, is unchanged). Frames: OUT_DIR/<kind>_<az>_<res>/NNNN.jpg (NNNN the kept frame's index, 20 per second,
real time), or OUT_DIR/frames/ when exactly one kind, view and resolution are asked (the hyde01 render job's layout)."""
import functools
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repository's root
from physmorph import gpu                                      # noqa: E402  (before torch)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw, ImageFont                    # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402
from physmorph.render.exterior import Lattice, make_field      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

dev = torch.device("cuda")
GREY = (.64, .69, .72)                                          # StudioRaster's albedo: the plain body
KINDS = ("A", "B", "B2", "C", "D", "U")


def srgb_to_linear(h):
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float64) / 255.0
    return tuple(np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4))


def ramp(stops):
    """A colour scale: (position in [0, 1], linear albedo) stops, linear in between."""
    pos = torch.tensor([s[0] for s in stops], device=dev)
    col = torch.tensor([s[1] for s in stops], device=dev, dtype=torch.float32)

    def f(t):
        t = t.clamp(0, 1)
        i = torch.searchsorted(pos, t.contiguous(), right=True).clamp(1, len(pos) - 1)
        w = ((t - pos[i - 1]) / (pos[i] - pos[i - 1])).clamp(0, 1)[:, None]
        return col[i - 1] * (1 - w) + col[i] * w
    return f


L = srgb_to_linear
NEUTRAL = L("#f0efec")
# one hue each (validated with the dataviz skill's validator, --ordinal; the diverging arms each as a ramp)
RAMP_A = ramp([(0, GREY), (.25, L("#f4c7a1")), (.5, L("#f0934f")), (.75, L("#e0601f")), (1, L("#8f2a08"))])
RAMP_B = ramp([(0, L("#dcd6f5")), (.25, L("#b0a6ee")), (.5, L("#8475df")), (.75, L("#5a49c4")), (1, L("#2e2270"))])
RAMP_C = ramp([(0, L("#9c1f1f")), (.125, L("#d93b3a")), (.25, L("#ec8a89")), (.375, L("#f6d0cf")), (.5, NEUTRAL),
               (.625, L("#d3e3f7")), (.75, L("#86b6ef")), (.875, L("#3987e5")), (1, L("#184f95"))])
RAMP_D = ramp([(0, GREY), (.25, L("#7fd1b0")), (.5, L("#1baf7a")), (.75, L("#0f7a54")), (1, L("#064532"))])
RAMP_U = ramp([(0, L("#184f95")), (.25, L("#5598e7")), (.5, NEUTRAL), (.75, L("#ec7a79")), (1, L("#b32222"))])
INK, INK2, INK3 = (236, 238, 240), (184, 191, 199), (160, 168, 178)
DARK, WHITE = (22, 26, 31), (250, 250, 248)


def options(argv):
    opts = dict(s[1:].split("=", 1) for s in argv[1:] if s.startswith("+"))
    return opts, [s for s in argv if not s.startswith("+")]


def pitch_of(s):
    return .708 * float(knn_self_torch(s, 9)[0][:, 8].median())


@functools.lru_cache(maxsize=16)
def font(size):
    """DejaVu Sans (it has the Greek and math signs the labels use): matplotlib's copy, else the system's."""
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]
    try:
        import matplotlib                                      # only for the location of its DejaVu font
        paths.insert(0, os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
    except ImportError:
        pass
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def rot_from_normal(n):
    rr = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    tt = nnf.normalize(torch.linalg.cross(n, rr), dim=1, eps=1e-9)
    return torch.stack((tt, torch.linalg.cross(n, tt), n), dim=2)


def lit_swatch(cam, albedo):
    """The colour a disc of this albedo shows facing the camera under the studio's lights (the colour bars' swatches):
    StudioRaster's shading for one front-facing normal at the image centre, tone map and sRGB as it applies them."""
    H, W = cam.to_camera.shape[:2]
    view = cam.to_camera[H // 2, W // 2]
    normal = view
    nv = float((normal * view).sum().clamp_min(1e-4))
    ambient = .20 + .12 * (.5 + .5 * float(normal[1]))
    rad = albedo * ambient
    roughness = .36
    alpha2, k = roughness ** 4, (roughness + 1) ** 2 / 8
    vis_v = nv / (nv * (1 - k) + k)
    for light, energy in cam.lights:
        hv = nnf.normalize(view + light, dim=0)
        nl = float((normal * light).sum().clamp(0, 1))
        nh = float((normal * hv).sum().clamp(0, 1))
        vh = float((view * hv).sum().clamp(0, 1))
        dist = alpha2 / (math.pi * max((nh * nh * (alpha2 - 1) + 1) ** 2, 1e-8))
        fres = .04 + .96 * (1 - vh) ** 5
        vis_l = nl / (nl * (1 - k) + k)
        spec = dist * fres * vis_v * vis_l / max(4 * nv * nl, 1e-5)
        rad = rad + (albedo * (1 - fres) / math.pi + spec) * energy * nl
    mapped = rad / (1 + rad)
    srgb = torch.where(mapped <= .0031308, 12.92 * mapped, 1.055 * mapped.clamp_min(0).pow(1 / 2.4) - .055)
    return (srgb.clamp(0, 1) * 255).round().byte().cpu().numpy()


def colour_bar(draw, cam, x0, y0, w, h, cmap, ticks, title, fnt):
    """A horizontal bar of the lit colours of the scale at positions t in [0, 1], tick marks and labels below (a label
    may hold a second line, drawn dimmer)."""
    t = torch.linspace(0, 1, w, device=dev)
    sw = lit_swatch(cam, cmap(t))
    for i in range(w):
        draw.line([(x0 + i, y0), (x0 + i, y0 + h)], fill=tuple(int(c) for c in sw[i]))
    draw.rectangle([x0 - 1, y0 - 1, x0 + w, y0 + h], outline=INK, width=1)
    for pos, lab in ticks:
        xx = x0 + pos * (w - 1)
        draw.line([(xx, y0 + h), (xx, y0 + h + max(4, h // 3))], fill=INK, width=max(1, h // 12))
        for j, part in enumerate(lab.split("\n")):
            tw = draw.textlength(part, font=fnt)
            draw.text((min(max(xx - tw / 2, x0 - 40), x0 + w + 40 - tw), y0 + h + max(6, h // 2) + j * (fnt.size + 3)),
                      part, font=fnt, fill=INK if j == 0 else INK2)
    for j, part in enumerate(reversed(title.split("\n"))):        # the title above the bar (lines upward)
        draw.text((x0, y0 - (j + 1) * (fnt.size + 4) - max(2, h // 5)), part, font=fnt, fill=INK)


def hatch_legend(draw, cam, x, y, sz, H, label, fnt):
    """The swatch of the hatched plain body (as drawn on the body) and its label."""
    w, h = int(46 * sz), int(22 * sz)
    base = tuple(int(v) for v in lit_swatch(cam, torch.tensor(GREY, device=dev)[None])[0])
    dark = tuple(int(.8 * v) for v in base)
    band = max(2, int(round(9 * H / 1080)) // 3)
    for yy in range(h):
        for xx in range(w):
            draw.point((x + xx, y + yy), fill=dark if ((xx + yy) // band) % 3 == 0 else base)
    draw.rectangle([x - 1, y - 1, x + w, y + h], outline=INK, width=1)
    draw.text((x + w + 12 * sz, y), label, font=fnt, fill=INK)


def wrap(draw, text, fnt, width):
    out, cur = [], ""
    for word in text.split(" "):
        test = (cur + " " + word).strip()
        if draw.textlength(test, font=fnt) <= width or not cur:
            cur = test
        else:
            out.append(cur)
            cur = word
    return out + ([cur] if cur else [])


def arrow(draw, p1, p2, kind, sz, alpha=None):
    """A 2D arrow along p1-p2: kind +1 stretch (double-headed, white with a dark outline), -1 compression (two heads
    meeting at the middle, dark with a white outline); alpha (0-255) on an RGBA layer."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    ln = float(np.hypot(*d))
    if ln < 6 * sz:
        return
    u = d / ln
    v = np.array([-u[1], u[0]])
    hl, hw = .32 * ln, .2 * ln
    fill, edge = (WHITE, DARK) if kind > 0 else (DARK, WHITE)
    if alpha is not None:
        fill, edge = fill + (alpha,), edge + (alpha,)
    lw = max(2, int(round(3.2 * sz)))
    if kind > 0:
        shafts = [(p1 + u * hl * .6, p2 - u * hl * .6)]
        heads = [(p1, u), (p2, -u)]                            # apex at the ends, pointing outward
    else:
        c = (p1 + p2) / 2
        g = .06 * ln
        shafts = [(p1, c - u * (g + hl * .6)), (p2, c + u * (g + hl * .6))]
        heads = [(c - u * g, -u), (c + u * g, u)]               # apexes at the middle, pointing inward
    tris = [[a, a + bk * hl + v * hw / 2, a + bk * hl - v * hw / 2] for a, bk in heads]
    for col, extra in ((edge, max(2, int(round(2.4 * sz)))), (fill, 0)):
        for s0, s1 in shafts:
            draw.line([tuple(s0), tuple(s1)], fill=col, width=lw + 2 * extra)
        for tri in tris:
            if extra:
                cen = sum(tri) / 3
                tri2 = [cen + (q - cen) * (1 + 2.2 * extra / max(hw, 1)) for q in tri]
                draw.polygon([tuple(q) for q in tri2], fill=col)
            else:
                draw.polygon([tuple(q) for q in tri], fill=col)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    opts, args = options(argv)
    kinds = [k for k in opts.get("viz", "A,B,C,D,U").split(",") if k in KINDS]
    views = [float(v) for v in opts.get("views", opts.get("az", "35")).split(",")]
    ress = [int(r) for r in opts.get("res", "1080").split(",")]
    EL, DIST = float(opts.get("el", 18.)), float(opts.get("dist", 1.05))
    FIELD, STRIDE = opts.get("field", "zbs"), int(opts.get("stride", 1))
    FRAMES = [int(v) for v in opts["frames"].split("-")] if "frames" in opts else None
    GCELL = float(opts.get("glyph_cell", 5.0))                               # pitches
    ARROW = float(opts.get("arrow_px", 46.0))                                # px at 1080 lines
    C_AVG = int(opts.get("c_avg", 1))                                       # C: running mean over this many windows
    D_AVG = int(opts.get("d_avg", 1))                                       # D: running mean over this many windows
    INTERP = opts.get("interp", "1") == "1"
    GB = "B" if "B" in kinds else "B2"                                     # the grid's B slot: B, else B2
    GRID = opts.get("grid", "1") == "1" and all(k in kinds for k in ("A", GB, "C", "D"))
    GNAME = "grid" if GB == "B" else "gridB2"
    GRID_FULL = opts.get("gridfull", "1") == "1"                            # 0: only the half-size 2 x 2 (4K panels)
    MESH = opts.get("mesh", Path(args[1]).name.split("300k")[0].split("_")[0])
    z = np.load(args[1], allow_pickle=False)
    out = Path(args[3])
    tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
    dense0 = torch.as_tensor(np.asarray(np.load(args[2].split(",")[0])["tgt"], np.float32), device=dev)
    a, a_c = pitch_of(tgt), pitch_of(dense0)
    del dense0
    center = tgt.mean(0)
    radius = float((tgt - center).norm(dim=1).max())
    h_lat = .92 * a_c
    lat = Lattice(center, 2.8 * radius)
    frames, raws = z["frames"], np.asarray(z["raws"]).astype(int)
    win_of = np.asarray(z["win_of_frame"]).astype(int)
    S = {k: np.asarray(z[k], np.float64) for k in ("w_lam", "w_t0", "w_t1", "w_layer_frac", "w_uactive_frac",
                                                   "w_graw_cos", "w_tot_rend", "w_tot_phys")}
    CLEAN = (np.asarray(z["w_cleanup_core_layer"], np.float64) if "w_cleanup_core_layer" in z.files
             else np.full(len(S["w_lam"]), np.nan))
    PHYS_ALL = "physics_def" in z.files and str(z["physics_def"]) == "all"
    sc = json.loads(str(z["scales"]))
    n_win = len(S["w_lam"])
    centres = 120.0 * (S["w_t0"] + S["w_t1"])                  # each window's centre in simulated frames (240 / s)
    A = {"rend": z["g_rend"], "phys": z["g_phys"], "ddfc": z["ddfc"], "du": z["du"], "layer": z["layer"],
         "ugate": z["ugate"], "dfc0": z["dfc0n"], "cos": z["cosrp"]}  # read once (an npz member is re-read per access)
    du_max, b_lo, b_hi = sc["v3_max"], sc["b_lo"], sc["b_hi"]
    # B2's reference: the outer layer's median |lambda g_render| per window, its running median over 5 windows
    medr = np.array([np.median(np.asarray(A["rend"][w])[np.asarray(A["layer"][w])]) for w in range(n_win)])
    MED5 = np.array([np.median(medr[max(0, w - 2):w + 3]) for w in range(n_win)])
    t_end = raws[-1] / 240.0
    print(f"{MESH}: pitch {a:.4f}, dense {a_c:.4f}; {len(raws)} kept frames, {n_win} windows; kinds {kinds} views {views} "
          f"res {ress}; interp {INTERP}; grid {GRID}; scales {sc}", flush=True)

    # the arrows' material subsample: one particle per occupied cell of a grid of GCELL pitches on the SOURCE positions
    x_src = torch.as_tensor(np.asarray(frames[0], np.float32), device=dev)
    cell = torch.floor((x_src - x_src.min(0).values) / (GCELL * a)).long()
    key = (cell[:, 0] * 100003 + cell[:, 1]) * 100003 + cell[:, 2]
    perm = torch.randperm(len(key), generator=torch.Generator(device="cpu").manual_seed(0)).to(dev)
    _, first = np.unique(key[perm].cpu().numpy(), return_index=True)
    G_IDX = perm[torch.as_tensor(first, device=dev)]
    G_RANK = torch.randperm(len(G_IDX), generator=torch.Generator(device="cpu").manual_seed(1)).to(dev)
    del x_src

    @functools.lru_cache(maxsize=3)
    def window(w):
        """A window's records on the device."""
        V = {ch: torch.as_tensor(np.asarray(A[ch][w], np.float32 if ch != "layer" else bool), device=dev) for ch in A}
        V["act"] = (V["layer"] & (V["ugate"] > .5)).float()  # where u acts (its gate is 0 or 1)
        V["ref_dfc0"] = torch.quantile(V["dfc0"][V["layer"]], .99)
        if C_AVG > 1:                                          # C: the running mean of the cosine over C_AVG windows
            lo_, hi_ = max(0, w - C_AVG // 2), min(n_win, w + C_AVG // 2 + 1)
            V["cos"] = torch.as_tensor(np.asarray(A["cos"][lo_:hi_], np.float32).mean(0), device=dev)
        if D_AVG > 1:                                          # D: running means of Delta dF_c and |dF_c(lambda = 0)|
            lo_, hi_ = max(0, w - D_AVG // 2), min(n_win, w + D_AVG // 2 + 1)
            V["ddfc"] = torch.as_tensor(np.asarray(A["ddfc"][lo_:hi_], np.float32).mean(0), device=dev)
            V["dfc0"] = torch.as_tensor(np.asarray(A["dfc0"][lo_:hi_], np.float32).mean(0), device=dev)
        return V

    def values(k):
        """The particles' values at kept frame k: linear between the centres of the windows around it (+interp=1), or
        the window's own (+interp=0); the nearer window (whose number and lambda the frame prints)."""
        r = float(raws[k])
        if INTERP:
            w0 = int(np.clip(np.searchsorted(centres, r, side="right") - 1, 0, n_win - 1))
            w1 = min(w0 + 1, n_win - 1)
            span = centres[w1] - centres[w0]
            al = 0.0 if w1 == w0 or span <= 0 else float(np.clip((r - centres[w0]) / span, 0, 1))
        else:
            w0 = w1 = int(win_of[k])
            al = 0.0
        V0, V1 = window(w0), window(w1)
        out = {ch: (1 - al) * V0[ch] + al * V1[ch] for ch in ("rend", "phys", "ddfc", "du", "act", "dfc0", "cos",
                                                               "ref_dfc0")}
        out["ref_b2"] = (1 - al) * MED5[w0] + al * MED5[w1]
        return out, (w0 if al < .5 else w1)

    def surface(x):
        """render_deliverable.Surface's base discs and normals (the deliverable's +fit=0 display)."""
        with torch.no_grad():
            p, g, _, _ = lat.discs(make_field(FIELD, x, a, radius=3., offset=.8, device_field=FIELD == "zbs"), h_lat)
        n = nnf.normalize(g, dim=1)
        d, nbx = knn_self_torch(x, 33)
        reach = float(d[(x - x[nbx[:, 1:]].mean(1)).norm(dim=1) >= .5 * a / .708, 32].median())
        spacing = float(knn_self_torch(p, 2)[0][:, 1].median())
        dd, nn = gpu.KNN(p).query(p, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
        wr = (dd.float() <= reach)[..., None]
        for _ in range(2):
            n = nnf.normalize((n[nn] * wr).sum(1), dim=1, eps=1e-9)
        return p, nnf.normalize(n, dim=1)

    cams, proj = {}, {}
    for az in views:
        for r in ress:
            W, H = r * 16 // 9, r
            cams[(az, r)] = StudioRaster(center, radius * DIST, W, H, az, EL)
            e, z_ = math.radians(EL), math.radians(az)          # the same camera, for the arrows' projection
            campos = center + 3.6 * radius * DIST * center.new_tensor((math.cos(e) * math.sin(z_), math.sin(e),
                                                                       math.cos(e) * math.cos(z_)))
            proj[(az, r)] = (world_to_view_torch(campos, center), campos, math.tan(math.radians(15)), W, H)
    single = len(kinds) * len(views) * len(ress) == 1
    dirs = {}
    for kd in kinds + (([GNAME] if GRID_FULL else []) + [GNAME + "half"] if GRID else []):
        for az in views:
            for r in ress:
                d = out / "frames" if single else out / f"{kd}_{int(az)}_{r}"
                d.mkdir(parents=True, exist_ok=True)
                dirs[(kd, az, r)] = d

    def splats(p, n):
        rot = rot_from_normal(n)
        sig = torch.tensor((h_lat ** 2, h_lat ** 2, (h_lat / 4) ** 2), device=dev)
        return (rot * sig[None, None]) @ rot.transpose(1, 2), torch.full((len(p),), .92, device=dev)

    def draw_body(cam, p, n, cov, op, alb, hatch=None):
        with torch.no_grad():
            img = cam(p, n, cov, op, normal_kernel=normal_filter_size(cam.to_camera.shape[0], False), albedo=alb)
            if hatch is not None and bool((hatch > 0).any()):  # a faint diagonal hatch where hatch (per disc) is 1
                cv = cam.raster_colors(p, n, cov, op, torch.ones_like(p))[..., 0]
                hm = cam.raster_colors(p, n, cov, op, hatch[:, None].expand(-1, 3).contiguous())[..., 0]
                frac = (hm / cv.clamp_min(1e-6)).clamp(0, 1) * (cv > .5)
                Hh, Wi = frac.shape
                band = max(2, int(round(9 * Hh / 1080)) // 3)
                yy, xx = torch.meshgrid(torch.arange(Hh, device=dev), torch.arange(Wi, device=dev), indexing="ij")
                img = img * (1 - .2 * (frac * (((xx + yy) // band) % 3 == 0).float()))[..., None]
        return (img.clamp(0, 1) * 255).round().byte().cpu().numpy()

    def project(key, q):
        view, _, ty, W, H = proj[key]
        qv = q @ view[:3, :3].T + view[:3, 3]
        zc = qv[:, 2].clamp_min(1e-6)
        col = ((qv[:, 0] / (zc * ty * W / H)) + 1) * W / 2 - .5
        row = (1 - qv[:, 1] / (zc * ty)) * H / 2 - .5
        return torch.stack((col, row), 1), qv[:, 2]

    def inset(dr, x0, y0, w, h, t_now, sz):
        """The total render and physics pulls over the morph (log), a cursor at the current time."""
        f = font(int(15 * sz))
        tt = .5 * (S["w_t0"] + S["w_t1"])
        lr, lp = np.log10(np.maximum(S["w_tot_rend"], 1e-30)), np.log10(np.maximum(S["w_tot_phys"], 1e-30))
        lo, hi = math.floor(min(lr.min(), lp.min())), math.ceil(max(lr.max(), lp.max()))
        X = lambda t: x0 + (t / max(t_end, 1e-9)) * w                       # noqa: E731
        Y = lambda v: y0 + h - (v - lo) / max(hi - lo, 1e-9) * h            # noqa: E731
        dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(120, 128, 138), width=1)
        for e in range(lo, hi + 1):
            dr.line([(x0, Y(e)), (x0 + w, Y(e))], fill=(92, 99, 108), width=1)
        for vals, col in ((lp, (190, 196, 204)), (lr, (240, 147, 79))):
            dr.line([(X(t), Y(v)) for t, v in zip(tt, vals)], fill=col, width=max(2, int(2 * sz)))
        dr.line([(X(t_now), y0), (X(t_now), y0 + h)], fill=WHITE, width=max(1, int(2 * sz)))
        dr.text((x0, y0 - f.size - 6 * sz), "total pull over the morph (log scale)", font=f, fill=INK)
        dr.text((x0 + w + 8 * sz, Y(lr[-1]) - f.size / 2), "render", font=f, fill=(240, 147, 79))
        dr.text((x0 + w + 8 * sz, Y(lp[-1]) - f.size / 2), "physics" if PHYS_ALL else "physics core", font=f,
                fill=(190, 196, 204))
        for tk in range(0, int(t_end) + 1, 2):
            dr.text((X(tk) - 4 * sz, y0 + h + 3 * sz), f"{tk}", font=f, fill=INK2)
        dr.text((x0 + w - 10 * sz, y0 + h + 3 * sz), "s", font=f, fill=INK2)

    def body_arrows(img, cam, key, p, n, cov, op, sites, sz):
        """D's arrows on the body, composited over it: at the subsample's sites facing the camera and not hidden (the
        depth of the drawn discs), ARROW px long on the screen times the axis' foreshortening (2 pitches along it
        against 2 across the view at that depth), each faded by its own fade and by the foreshortening over 0.3-0.45,
        thinned to one per 1.15 lengths in a fixed order of the particles (no popping as the values change)."""
        gq_, gn_, et_, sg_, fd_, rk_ = sites
        if not len(gq_):
            return img
        Wd, Hd = img.size
        _, campos, _, _, _ = proj[key]
        _, zq = project(key, p)
        zmin, zmax = float(zq.min()), float(zq.max())
        zc = ((zq - zmin) / max(zmax - zmin, 1e-9))[:, None].expand(-1, 3).contiguous()
        cv = cam.raster_colors(p, n, cov, op, torch.ones_like(p))[..., 0]
        depth = cam.raster_colors(p, n, cov, op, zc)[..., 0] / cv.clamp_min(1e-6)
        c2, zg = project(key, gq_)
        q1, _ = project(key, gq_ - a * et_)
        q2, _ = project(key, gq_ + a * et_)
        dq = q2 - q1
        full = 2 * a * (Hd / (2 * math.tan(math.radians(15)))) / zg.clamp_min(1e-6)
        fore = (dq.norm(dim=1) / full).clamp(max=1.0)
        u2 = dq / dq.norm(dim=1, keepdim=True).clamp_min(1e-9)
        half = .5 * ARROW * sz * fore
        e1, e2 = c2 - u2 * half[:, None], c2 + u2 * half[:, None]
        ci = c2.round().long()
        inb = (ci[:, 0] >= 0) & (ci[:, 0] < Wd) & (ci[:, 1] >= 0) & (ci[:, 1] < Hd)
        ci = ci.clamp_min(0)
        ci[:, 0], ci[:, 1] = ci[:, 0].clamp(max=Wd - 1), ci[:, 1].clamp(max=Hd - 1)
        zn = (zg - zmin) / max(zmax - zmin, 1e-9)
        vis = ((((campos - gq_) * gn_).sum(1) > 0) & inb & (cv[ci[:, 1], ci[:, 0]] > .5)
               & (zn <= depth[ci[:, 1], ci[:, 0]] + 1.0 * a / max(zmax - zmin, 1e-9)))
        alpha = fd_ * ((fore - .3) / .15).clamp(0, 1) * vis.float()
        over = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
        do = ImageDraw.Draw(over)
        taken = []
        for q in torch.argsort(rk_).tolist():
            if float(alpha[q]) <= 0:
                continue
            cxy = c2[q].cpu().numpy()
            if any((cxy[0] - t0) ** 2 + (cxy[1] - t1) ** 2 < (1.15 * ARROW * sz) ** 2 for t0, t1 in taken):
                continue
            taken.append((cxy[0], cxy[1]))
            arrow(do, e1[q].cpu().numpy(), e2[q].cpu().numpy(), float(sg_[q]), sz, alpha=int(255 * float(alpha[q])))
        return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")

    def phys_note(w):
        """What "physics" means in A and C: every non-render term (rgv_patch2 dumps), or the core alone."""
        if PHYS_ALL:
            return ("physics = all non-render terms: the core (transport, surface proximity, end drift; released "
                    "motion) and the cleanup (spray cleanup, near band)")
        r = CLEAN[w]
        return ("physics = the physics core (transport, surface proximity, end drift; released motion); the cleanup "
                "terms (spray cleanup, near band) act separately, the spray cleanup mainly on detached particles"
                + (f"; on the outer layer now their pull is {r:.2f} of the core's" if np.isfinite(r) else ""))

    grey = torch.tensor(GREY, device=dev, dtype=torch.float32)
    keep = list(range(0, len(raws), STRIDE)) + ([len(raws) - 1] if (len(raws) - 1) % STRIDE else [])
    if FRAMES is not None:
        keep = [k for k in keep if FRAMES[0] <= k <= FRAMES[1]]
    tnote = ("values interpolated in time between window centres (a window = 20 driven + 20 released steps, 1/6 s)"
             if INTERP else "values held through each window (raw; a window = 20 driven + 20 released steps, 1/6 s)")
    for i, k in enumerate(keep):
        V, wn = values(k)
        x = torch.as_tensor(np.asarray(frames[k], np.float32), device=dev)
        p, n = surface(x)
        cov, op = splats(p, n)
        tx = gpu.KNN(x)
        dd, ii = tx.query(p, 8)
        wgt = torch.exp(-(dd.float() / a) ** 2)
        wgt = wgt / wgt.sum(1, keepdim=True).clamp_min(1e-12)

        def at_discs(v):
            return (wgt * v[ii]).sum(1)
        r_d, p_d = at_discs(V["rend"]), at_discs(V["phys"])
        share = r_d / (r_d + p_d).clamp_min(1e-30)
        alb, hatch = {}, {}
        alb["A"] = RAMP_A(share)
        alb["B"] = RAMP_B((torch.log10(r_d.clamp_min(1e-30)) - math.log10(b_lo)) / (math.log10(b_hi) - math.log10(b_lo)))
        alb["B2"] = RAMP_B(.5 + .5 * torch.log10((r_d / V["ref_b2"]).clamp_min(1e-30)))   # 1/10x .. 10x the typical
        hatch["C"] = (share < .05).float()
        alb["C"] = torch.where(hatch["C"][:, None] > 0, grey, RAMP_C(.5 + .5 * at_discs(V["cos"]).clamp(-1, 1)))
        dn_d, d0_d = at_discs(V["ddfc"].norm(dim=1)), at_discs(V["dfc0"])
        hatch["D"] = (d0_d < .01 * V["ref_dfc0"]).float()
        alb["D"] = torch.where(hatch["D"][:, None] > 0, grey, RAMP_D((dn_d / d0_d.clamp_min(1e-30)).clamp(0, 1)))
        if "U" in kinds:
            cov3 = at_discs(V["act"])
            v3 = at_discs(V["act"] * V["du"]) / cov3.clamp_min(1e-12)
            on3 = ((cov3 - .2) / .3).clamp(0, 1)
            alb["U"] = on3[:, None] * RAMP_U(.5 + .5 * v3 / du_max) + (1 - on3[:, None]) * grey
            hatch["U"] = 1 - on3
        if "D" in kinds:                                       # the arrows' sites: the subsample now at the surface
            dsd, js = gpu.KNN(p).query(x[G_IDX], 1)
            gsel = dsd.float().reshape(-1) < 1.0 * a
            js = js.reshape(-1).long()[gsel]
            gq, gn = p[js], n[js]
            d8, i8 = tx.query(gq, 8)
            w8 = torch.exp(-(d8.float() / a) ** 2)
            w8 = w8 / w8.sum(1, keepdim=True).clamp_min(1e-12)
            M = (w8[..., None] * V["ddfc"][i8]).sum(1).view(-1, 3, 3)
            relg = (w8 * V["ddfc"][i8].norm(dim=2)).sum(1) / (w8 * V["dfc0"][i8]).sum(1).clamp_min(1e-30)
            d0g = (w8 * V["dfc0"][i8]).sum(1)
            ev, evec = torch.linalg.eigh(.5 * (M + M.transpose(1, 2)))
            j = ev.abs().argmax(1)
            lam = ev.gather(1, j[:, None]).squeeze(1)
            e = evec.gather(2, j[:, None, None].expand(-1, 3, 1)).squeeze(2)
            et = e - (e * gn).sum(1, keepdim=True) * gn
            # each arrow fades in and out (no popping): over a change of 15-25 % (drawn "over 20 %") and over a
            # tangent part of the axis of 0.2-0.4; its place in the thinning is a fixed rank of its particle
            fade = ((relg - .15) / .1).clamp(0, 1) * ((et.norm(dim=1) - .2) / .2).clamp(0, 1)
            ok = (fade > 0) & (d0g >= .01 * V["ref_dfc0"])
            sites = (gq[ok] + .3 * a * gn[ok], gn[ok], nnf.normalize(et[ok], dim=1), torch.sign(lam[ok]), fade[ok],
                     G_RANK[gsel][ok])
        lam_w = S["w_lam"][wn]
        info = f"t = {raws[k] / 240.0:5.2f} s   ·   window {wn + 1} of {n_win}   ·   λ = {lam_w:.3g}"
        pcg = S["w_graw_cos"][wn]
        for az in views:
            for r in ress:
                sz = r / 1080.0
                f_t, f_s, f_p = font(int(32 * sz)), font(int(20 * sz)), font(int(16 * sz))
                cam = cams[(az, r)]
                panels = {}
                for kd in kinds:
                    img = Image.fromarray(draw_body(cam, p, n, cov, op, alb[kd], hatch.get(kd)))
                    if kd == "D":                              # the arrows on the body first, under the legends
                        img = body_arrows(img, cam, (az, r), p, n, cov, op, sites, sz)
                    dr = ImageDraw.Draw(img)
                    Wd, Hd = img.size
                    bx, by, bw = int(Wd - 640 * sz), int(Hd - 130 * sz), int(520 * sz)
                    small = [tnote]
                    if kd == "A":
                        title = "Where does the render act on dFᴄ?"
                        colour_bar(dr, cam, bx, by, bw, int(20 * sz), RAMP_A,
                                   [(0, "0 %\ngrey: physics only"), (.25, "25 %"), (.5, "50 %"), (.75, "75 %"),
                                    (1, "100 %\norange: render dominates")],
                                   "the render's share of the pull on this particle's control", f_s)
                        small = ["share = |λ g_render| / (|λ g_render| + |g_phys|), g = ∂L/∂dFᴄ of the window's first "
                                 "gradient summed over its 20 driven steps (the render's after PCGrad, times λ)"] + small
                        small = [phys_note(wn)] + small
                    elif kd == "B":
                        title = "How strong is the render's pull?"
                        colour_bar(dr, cam, bx, by, bw, int(20 * sz), RAMP_B,
                                   [(0, "weak"), (.5, "medium"), (1, "strong")],
                                   "the render's pull on this particle's control\n(one scale for the whole run)", f_s)
                        inset(dr, int(40 * sz), int(Hd - 330 * sz), int(470 * sz), int(110 * sz), raws[k] / 240.0, sz)
                        small = [f"|λ ∂L_render/∂dFᴄ| summed over the window's 20 driven steps, log scale: weak = "
                                 f"{b_lo:.1g}, medium = {math.sqrt(b_lo * b_hi):.1g}, strong = {b_hi:.1g} (objective "
                                 f"units); λ is reset every window so that the render's total is half the physics' total "
                                 f"over all controls, and both fade as the body arrives"] + small
                    elif kd == "B2":
                        title = "How strong is the render's pull here, against its typical pull now?"
                        colour_bar(dr, cam, bx, by, bw, int(20 * sz), RAMP_B,
                                   [(0, "1/10×\nweaker than usual now"), (.5, "1×\ntypical"),
                                    (1, "10×\nstronger than usual now")],
                                   "the render's pull on this particle's control,\nagainst the typical pull at this moment", f_s)
                        inset(dr, int(40 * sz), int(Hd - 330 * sz), int(470 * sz), int(110 * sz), raws[k] / 240.0, sz)
                        small = [f"|λ ∂L_render/∂dFᴄ| summed over the window's 20 driven steps, divided by the typical "
                                 f"pull now: the outer layer's median, a running median over 5 windows (0.83 s), "
                                 f"interpolated in time ({float(V['ref_b2']):.2g}); the absolute fade is in the inset"] + small
                    elif kd == "C":
                        title = "Where do the render and the physics disagree?"
                        colour_bar(dr, cam, bx, by, bw, int(20 * sz), RAMP_C,
                                   [(0, "−1\nred: against each other"), (.5, "0\nunrelated"),
                                    (1, "+1\nblue: the same way")],
                                   ("do the render and the physics pull this particle's\ncontrol the same way? "
                                    f"(averaged over {C_AVG} windows, {C_AVG / 6:.2f} s)" if C_AVG > 1 else
                                    "do the render and the physics pull\nthis particle's control the same way?"), f_s)
                        hatch_legend(dr, cam, bx, int(Hd - 228 * sz), sz, Hd,
                                     "render barely acts here (share under 5 %)", f_s)
                        small = ["cosine at this particle between λ ∂L_render/∂dFᴄ (before PCGrad) and ∂L_phys/∂dFᴄ, "
                                 "9 components, summed over the window's 20 driven steps",
                                 "PCGrad acts on all controls at once: it removes the conflicting part of the render "
                                 "gradient only in windows where the two disagree overall (this window: "
                                 f"{'yes' if pcg < 0 else 'no'})"] + small
                        small = [phys_note(wn)] + small
                        if C_AVG > 1:
                            small = [f"the cosine is a running mean over {C_AVG} windows ({C_AVG / 6:.2f} s): per particle it changes "
                                     f"much from one window to the next"] + small
                    elif kd == "D":
                        title = "What did the render do to dFᴄ?"
                        colour_bar(dr, cam, bx, by, bw, int(20 * sz), RAMP_D,
                                   [(0, "0 %"), (.25, "25 %"), (.5, "50 %"), (.75, "75 %"), (1, "≥ 100 %")],
                                   "how much the render changed this particle's control\n(% of it without the render"
                                   + (f"; {D_AVG}-window mean, {D_AVG / 6:.2f} s)" if D_AVG > 1 else ")"), f_s)
                        hatch_legend(dr, cam, bx, int(Hd - 228 * sz), sz, Hd,
                                     "control negligible here (under 1 % of the largest)", f_s)
                        ax, ay = bx, int(Hd - 318 * sz)
                        arrow(dr, (ax, ay + 12 * sz), (ax + 60 * sz, ay + 12 * sz), 1, sz)
                        dr.text((ax + 76 * sz, ay), "the render's change stretches along this line", font=f_s, fill=INK)
                        arrow(dr, (ax, ay + 50 * sz), (ax + 60 * sz, ay + 50 * sz), -1, sz)
                        dr.text((ax + 76 * sz, ay + 38 * sz), "it compresses along this line", font=f_s, fill=INK)
                        small = ["the same window optimised twice from the same start, with the render and with λ = 0; "
                                 "colour |ΔdFᴄ| / |dFᴄ(λ = 0)| of the 20-step means; arrows (only where the change "
                                 "is over 20 %): the principal axis of sym(ΔdFᴄ) along the surface; dFᴄ is added "
                                 "to F every driven step"] + small
                    else:
                        title = "How does the render change the outer layer's normal offset u?"
                        tk = [(0.0, f"−{du_max:.2g}\ninward"), (.25, f"−{du_max / 2:.2g}"), (.5, "0"),
                              (.75, f"{du_max / 2:.2g}"), (1.0, f"{du_max:.2g}\noutward")]
                        colour_bar(dr, cam, bx, by, bw, int(20 * sz), RAMP_U, tk,
                                   "how far the render moved the layer along its normal (pitches)", f_s)
                        hatch_legend(dr, cam, bx, int(Hd - 228 * sz), sz, Hd,
                                     "u does not act here (outside the layer, or gate closed)", f_s)
                        small = [f"g Δu = g (u(render on) − u(λ = 0)), same window, same start; the outer layer is "
                                 f"{100 * S['w_layer_frac'][wn]:.1f} % of the particles, u acts on "
                                 f"{100 * S['w_uactive_frac'][wn]:.0f} % of it (where the body has arrived)"] + small
                    dr.text((40 * sz, 30 * sz), title, font=f_t, fill=INK)
                    lines = []
                    for s_ in small:
                        lines += wrap(dr, s_, f_p, 1000 * sz)
                    yb = Hd - (34 + 22 * len(lines)) * sz
                    dr.text((40 * sz, yb - 34 * sz), info, font=f_s, fill=INK)
                    for j2, s_ in enumerate(lines):
                        dr.text((40 * sz, yb + j2 * 22 * sz), s_, font=f_p, fill=INK2)
                    dr.text((40 * sz, Hd - 28 * sz), f"{MESH} 300k, seed 97, deliverable code; 1 pitch = {a:.4f} wu",
                            font=font(int(15 * sz)), fill=INK3)
                    img.save(dirs[(kd, az, r)] / f"{k:04d}.jpg", quality=93)
                    panels[kd] = img
                if GRID:
                    Wd, Hd = panels["A"].size
                    g = Image.new("RGB", (2 * Wd, 2 * Hd))
                    for kd, (ox, oy) in zip(("A", GB, "C", "D"), ((0, 0), (Wd, 0), (0, Hd), (Wd, Hd))):
                        g.paste(panels[kd], (ox, oy))
                    ImageDraw.Draw(g).line([(Wd, 0), (Wd, 2 * Hd)], fill=(40, 45, 52), width=max(2, int(3 * sz)))
                    ImageDraw.Draw(g).line([(0, Hd), (2 * Wd, Hd)], fill=(40, 45, 52), width=max(2, int(3 * sz)))
                    if GRID_FULL:
                        g.save(dirs[(GNAME, az, r)] / f"{k:04d}.jpg", quality=93)
                    g.resize((Wd, Hd), Image.LANCZOS).save(dirs[(GNAME + "half", az, r)] / f"{k:04d}.jpg", quality=93)
        if i % 20 == 0 or k == keep[-1]:
            print(json.dumps(dict(frame=k, raw=int(raws[k]), window=wn + 1, discs=len(p),
                                  arrow_sites=int(len(sites[0])) if "D" in kinds else 0)), flush=True)
    print("wrote", out, len(keep), "frames", flush=True)


if __name__ == "__main__":
    main()
