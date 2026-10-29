"""ri_video.py MESH VALS_DIR OUT_DIR [+res=1080] [+views=35,90] [+dmax=0.75] [+frames=a-b] [+stride=1] [+grid=1]
-- render influence video, pass 2 (one card): "Where does the render change the shape?" drawn on the full arm's
deforming body in the RGV look (scripts/render_gradient_viz.py of repo_rgvtag: its display surface, discs, splats,
camera, lights, backdrop, fonts, colour bar and text layout, imported, not copied).

Value per particle (pass 1, ri_video_values.py): d = how far the full arm's surface ("dF_c + layer u + render loss
(full)", L97) lies outside (+) or inside (-) the twin's ("dF_c + layer u", lambda = 0, P97) at the same simulated time,
pitches, averaged over the surface around it (1.5 pitches); e2 = the same arm run twice (L97 vs RG97), squared and
averaged over 2 pitches: the run-to-run noise there. Here both are averaged over +-2 kept frames (+-0.1 s) by particle,
carried to the display discs as RGV does (the Gaussian-weighted mean of a disc's 8 nearest particles), and the colour is
grey where |d| < 1.5 sqrt(e2), the full colour from 2.5 sqrt(e2) (a soft step: no popping), blue = pulled in, red =
pushed out, one fixed scale (+dmax) for the run. The body image is flipped left-right before the text is drawn (the
studio camera mirrors). Grid (+grid=1): top the plain full arm and twin (front), bottom the map front and side."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

B = Path("/data/relcfd/chayo/physmorph_v2")
spec = importlib.util.spec_from_file_location("rgv", B / "repo_rgvtag/scripts/render_gradient_viz.py")
RGV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RGV)                                   # its module level: the repo on the path, dev = cuda
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph import gpu                                      # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402
from physmorph.render.exterior import Lattice                  # noqa: E402

dev = RGV.dev
L_ = RGV.L
GREY = RGV.GREY
RAMP_E = RGV.ramp([(0, L_("#184f95")), (.25, L_("#5598e7")), (.5, GREY), (.75, L_("#ec7a79")), (1, L_("#b32222"))])
ORANGE, LGREY = (240, 147, 79), (190, 196, 204)
G = B / "output/gpu"


def main():
    opts, args = RGV.options(sys.argv)
    mesh, vdir, out = args[1], Path(args[2]), Path(args[3])
    views = [float(v) for v in opts.get("views", "35,90").split(",")]
    res = int(opts.get("res", 1080))
    DMAX = float(opts.get("dmax", 0.75))
    STRIDE = int(opts.get("stride", 1))
    GRID = opts.get("grid", "1") == "1"
    EL, DIST = 18.0, 1.05
    meta = json.load(open(vdir / f"vals_{mesh}.json"))
    rows = meta["rows"]
    D = np.load(vdir / f"vals_{mesh}_d.npy", mmap_mode="r")
    E2 = np.load(vdir / f"vals_{mesh}_e2.npy", mmap_mode="r")
    zL = np.load(G / f"d145t_b1/{mesh}300k_L97_frames12.npz", allow_pickle=True)
    FL = zL["frames"]
    zP = np.load(G / f"d145t_b1/{mesh}300k_P97_frames12.npz", allow_pickle=True)
    FP = zP["frames"]
    hist = [h for h in next(iter(json.load(open(G / f"d145t_b1/{mesh}300k_L97.json"))["arms"].values()))["history"]
            if "animation" in h]
    lam = [h.get("lambda") for h in hist]
    n_win = len(hist)
    win = json.loads(next(ln for ln in open(G / "d159/read/window.jsonl") if f'"{mesh}_tag_L97"' in ln))
    t_arr = (win["arrival_window"] + 1) / 6.0
    tgt = torch.as_tensor(np.asarray(zL["tgt"], np.float32), device=dev)
    a = RGV.pitch_of(tgt)
    center = tgt.mean(0)
    radius = float((tgt - center).norm(dim=1).max())
    # the deliverable's display: the dense sample's pitch sets the lattice (RGV's DENSE_NPZ, d111's 1.5M sample)
    dense = torch.as_tensor(np.asarray(np.load(G / f"d111/{mesh}_dense15.npz")["tgt"], np.float32), device=dev)
    a_c = RGV.pitch_of(dense)
    del dense
    h_lat = .92 * a_c
    lat = Lattice(center, 2.8 * radius)
    K = len(rows)
    keep = list(range(0, K, STRIDE)) + ([K - 1] if (K - 1) % STRIDE else [])
    if "frames" in opts:
        lo, hi = (int(v) for v in opts["frames"].split("-"))
        keep = [k for k in keep if lo <= k <= hi]
    W, H = res * 16 // 9, res
    sz = res / 1080.0
    cams = {az: StudioRaster(center, radius * DIST, W, H, az, EL) for az in views}
    t_end = rows[-1]["t"]
    tt = np.array([r["t"] for r in rows])
    rd = np.array([r["rms_d"] for r in rows])
    re_ = np.array([r["rms_e"] for r in rows])
    t_R_end = meta["rawR_end"] / 240.0
    t_P_end = meta["rawP_end"] / 240.0
    HERO = {int(np.argmin(np.abs(tt - t_))) for t_ in (0.5, 1.0, 2.0)} | {K - 1}
    dirs = {}
    for kd in [f"map_{int(az)}" for az in views] + (["grid", "gridhalf"] if GRID else []) + ["hero"]:
        dirs[kd] = out / mesh / kd
        dirs[kd].mkdir(parents=True, exist_ok=True)
    print(f"{mesh}: pitch {a:.4f}, {K} kept frames, {len(keep)} drawn, windows {n_win}, dmax {DMAX}, arrival {t_arr:.2f} s",
          flush=True)

    def surface(x):
        """RGV's display surface (the deliverable's +fit=0 base display) for these particles."""
        with torch.no_grad():
            from physmorph.render.exterior import make_field
            import torch.nn.functional as nnf
            p, g, _, _ = lat.discs(make_field("zbs", x, a, radius=3., offset=.8, device_field=True), h_lat)
            n = nnf.normalize(g, dim=1)
            d, nbx = RGV.knn_self_torch(x, 33)
            reach = float(d[(x - x[nbx[:, 1:]].mean(1)).norm(dim=1) >= .5 * a / .708, 32].median())
            spacing = float(RGV.knn_self_torch(p, 2)[0][:, 1].median())
            dd, nn = gpu.KNN(p).query(p, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
            wr = (dd.float() <= reach)[..., None]
            for _ in range(2):
                n = nnf.normalize((n[nn] * wr).sum(1), dim=1, eps=1e-9)
            return p, nnf.normalize(n, dim=1)

    def splats(p, n):
        rot = RGV.rot_from_normal(n)
        sig = torch.tensor((h_lat ** 2, h_lat ** 2, (h_lat / 4) ** 2), device=dev)
        return (rot * sig[None, None]) @ rot.transpose(1, 2), torch.full((len(p),), .92, device=dev)

    def body(cam, p, n, alb):
        cov, op = splats(p, n)
        with torch.no_grad():
            img = cam(p, n, cov, op, normal_kernel=normal_filter_size(cam.to_camera.shape[0], False), albedo=alb)
        arr = (img.clamp(0, 1) * 255).round().byte().cpu().numpy()
        return Image.fromarray(np.ascontiguousarray(arr[:, ::-1]))     # undo the studio camera's mirror

    def tmean(arr, k):
        lo_, hi_ = max(0, k - 2), min(K, k + 3)
        blk = np.asarray(arr[lo_:hi_], np.float32)
        cnt = np.isfinite(blk).sum(0)
        s = np.nansum(blk, 0)
        v = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        return torch.as_tensor(v, dtype=torch.float32, device=dev)

    def inset(dr, x0, y0, w, h, t_now):
        f = RGV.font(int(15 * sz))
        hi = max(rd.max(), re_.max()) * 1.1
        X = lambda t: x0 + (t / max(t_end, 1e-9)) * w                      # noqa: E731
        Y = lambda v: y0 + h - v / hi * h                                   # noqa: E731
        dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(120, 128, 138), width=1)
        step = 0.25 if hi < 0.6 else 0.5
        v = step
        while v < hi:
            dr.line([(x0, Y(v)), (x0 + w, Y(v))], fill=(92, 99, 108), width=1)
            dr.text((x0 - 34 * sz, Y(v) - f.size / 2), f"{v:g}", font=f, fill=RGV.INK2)
            v += step
        poly = [(X(t), Y(0)) for t in tt[:1]] + [(X(t), Y(e)) for t, e in zip(tt, re_)] + [(X(tt[-1]), Y(0))]
        dr.polygon(poly, fill=(104, 112, 122))
        dr.line([(X(t), Y(v)) for t, v in zip(tt, rd)], fill=ORANGE, width=max(2, int(2 * sz)))
        dr.line([(X(t_arr), y0), (X(t_arr), y0 + h)], fill=(150, 158, 168), width=1)
        dr.line([(X(t_now), y0), (X(t_now), y0 + h)], fill=RGV.WHITE, width=max(1, int(2 * sz)))
        dr.text((x0, y0 - 2 * f.size - 8 * sz), "how far apart the two surfaces are, whole body (rms, pitches)",
                font=f, fill=RGV.INK)
        dr.text((x0, y0 - f.size - 4 * sz), "full vs twin", font=f, fill=ORANGE)
        dr.text((x0 + dr.textlength("full vs twin   ", font=f), y0 - f.size - 4 * sz),
                "the same full arm run twice (noise)", font=f, fill=LGREY)
        dr.text((X(t_arr) + 4 * sz, y0 + 2 * sz), "arrival", font=f, fill=RGV.INK2)
        for tk in range(0, int(t_end) + 1, 2 if t_end < 20 else 5):
            if X(tk) + 20 * sz > x0 + w - 10 * sz:
                continue
            dr.text((X(tk) - 4 * sz, y0 + h + 3 * sz), f"{tk}", font=f, fill=RGV.INK2)
        dr.text((x0 + w - 10 * sz, y0 + h + 3 * sz), "s", font=f, fill=RGV.INK2)

    def swatch(dr, cam, x, y, label, fnt):
        w, h = int(46 * sz), int(22 * sz)
        col = tuple(int(v) for v in RGV.lit_swatch(cam, torch.tensor(GREY, device=dev)[None])[0])
        dr.rectangle([x, y, x + w, y + h], fill=col, outline=RGV.INK, width=1)
        dr.text((x + w + 12 * sz, y), label, font=fnt, fill=RGV.INK)

    def text_frame(img, cam, title, info, small, bar=True, t_now=None):
        dr = ImageDraw.Draw(img)
        f_t, f_s, f_p = RGV.font(int(32 * sz)), RGV.font(int(20 * sz)), RGV.font(int(16 * sz))
        Wd, Hd = img.size
        dr.text((40 * sz, 30 * sz), title, font=f_t, fill=RGV.INK)
        if bar:
            bx, by, bw = int(Wd - 640 * sz), int(Hd - 130 * sz), int(520 * sz)
            RGV.colour_bar(dr, cam, bx, by, bw, int(20 * sz), RAMP_E,
                           [(0, f"−{DMAX:g}\nrender pulled it in"), (.25, f"−{DMAX / 2:g}"), (.5, "0"),
                            (.75, f"+{DMAX / 2:g}"), (1, f"+{DMAX:g}\nrender pushed it out")],
                           "how far the render moved the surface here\n(full arm against its twin at the same moment, pitches)",
                           f_s)
            swatch(dr, cam, bx, int(Hd - 228 * sz), "grey: within the run-to-run noise here", f_s)
            inset(dr, int(40 * sz), int(Hd - 360 * sz), int(470 * sz), int(110 * sz), t_now)
        lines = []
        for s_ in small:
            lines += RGV.wrap(dr, s_, f_p, 1000 * sz)
        yb = Hd - (34 + 22 * len(lines)) * sz
        dr.text((40 * sz, yb - 34 * sz), info, font=f_s, fill=RGV.INK)
        for j, s_ in enumerate(lines):
            dr.text((40 * sz, yb + j * 22 * sz), s_, font=f_p, fill=RGV.INK2)
        dr.text((40 * sz, Hd - 28 * sz), f"{mesh} 300k, seed 97, code freeze-2026-10-15; 1 pitch = {a:.4f} wu; "
                f"image flipped left-right (the studio camera mirrors)", font=RGV.font(int(15 * sz)), fill=RGV.INK3)
        return img

    small_map = [
        "colour: how far the surface of \"dFᴄ + layer u + render loss (full)\" lies outside (red) or inside (blue) the "
        "surface of its twin \"dFᴄ + layer u\" (λ = 0) at the same simulated time; the exterior field the loss reads "
        "(radius 3 pitches), averaged over 1.5 pitches and ±0.1 s",
        "grey where the change is under 1.5–2.5 × the run-to-run noise at that spot (the full arm run twice at seed "
        f"97, its square averaged over 2 pitches); after the re-run ends ({t_R_end:.1f} s) its last frame is used"
        + (f", after the twin's run ends ({t_P_end:.1f} s) its last frame" if t_P_end < t_end else ""),
        f"λ: the render's weight in this window (λ|g_render| = |g_physics| / 2 by the rule); arrival (90 % of the "
        f"motion done) at {t_arr:.1f} s"]
    for i, k in enumerate(keep):
        r = rows[k]
        x = torch.as_tensor(np.asarray(FL[k], np.float32), device=dev)
        p, n = surface(x)
        dv, e2 = tmean(D, k), tmean(E2, k)
        tx = gpu.KNN(x)
        dd, ii = tx.query(p, 8)
        wgt = torch.exp(-(dd.float() / a) ** 2)
        ok_d = torch.isfinite(dv)[ii].float()
        wd = wgt * ok_d
        d_d = (wd * torch.nan_to_num(dv)[ii]).sum(1) / wd.sum(1).clamp_min(1e-12)
        ok_e = torch.isfinite(e2)[ii].float()
        we = wgt * ok_e
        s_d = ((we * torch.nan_to_num(e2)[ii]).sum(1) / we.sum(1).clamp_min(1e-12)).clamp_min(1e-6).sqrt()
        has = (wd.sum(1) > 1e-9).float()
        wcol = ((d_d.abs() / s_d - 1.5) / 1.0).clamp(0, 1) * has
        g = torch.tensor(GREY, device=dev, dtype=torch.float32)
        alb = g + wcol[:, None] * (RAMP_E(.5 + .5 * d_d / DMAX) - g)
        w = min(max(r["raw"] - 1, 0) // 40, n_win - 1)
        lw = lam[w]
        info = f"t = {r['t']:5.2f} s   ·   window {w + 1} of {n_win}   ·   λ = {lw:.3g}" if lw is not None else \
            f"t = {r['t']:5.2f} s   ·   window {w + 1} of {n_win}   ·   window nulled"
        panels = {}
        for az in views:
            cam = cams[az]
            bimg = body(cam, p, n, alb)
            if k in HERO:                                      # the hero sheet's bodies, without the text
                bimg.save(dirs["hero"] / f"{k:04d}_{int(az)}.png")
            img = text_frame(bimg, cam, "Where does the render change the shape?", info, small_map, t_now=r["t"])
            img.save(dirs[f"map_{int(az)}"] / f"{k:04d}.jpg", quality=93)
            panels[az] = img
        if GRID:
            cam = cams[views[0]]
            g_ = torch.tensor(GREY, device=dev, dtype=torch.float32)
            full = text_frame(body(cam, p, n, None), cam, "dFᴄ + layer u + render loss (full)",
                              f"t = {r['t']:5.2f} s", ["the delivered run L97 (plain studio look)"], bar=False)
            xp = torch.as_tensor(np.asarray(FP[r["kp"]], np.float32), device=dev)
            pp, npp = surface(xp)
            twin = text_frame(body(cam, pp, npp, None), cam, "dFᴄ + layer u (twin, λ = 0)",
                              f"t = {r['rawP'] / 240.0:5.2f} s",
                              ["the same run without the render loss (P97, plain studio look)"
                               + ("; its run has ended, its last frame" if r["rawP"] < r["raw"] else "")], bar=False)
            del g_
            gr = Image.new("RGB", (2 * W, 2 * H))
            for im, (ox, oy) in zip((full, twin, panels[views[0]], panels[views[-1]]), ((0, 0), (W, 0), (0, H), (W, H))):
                gr.paste(im, (ox, oy))
            dr = ImageDraw.Draw(gr)
            dr.line([(W, 0), (W, 2 * H)], fill=(40, 45, 52), width=max(2, int(3 * sz)))
            dr.line([(0, H), (2 * W, H)], fill=(40, 45, 52), width=max(2, int(3 * sz)))
            gr.save(dirs["grid"] / f"{k:04d}.jpg", quality=93)
            gr.resize((W, H), Image.LANCZOS).save(dirs["gridhalf"] / f"{k:04d}.jpg", quality=93)
        if i % 20 == 0 or k == keep[-1]:
            print(json.dumps(dict(frame=k, t=r["t"], window=w + 1, discs=len(p), coloured=float((wcol > .5).float().mean()),
                                  d_p99=float(torch.quantile(d_d.abs(), .99)))), flush=True)
    print("wrote", out / mesh, len(keep), "frames", flush=True)


if __name__ == "__main__":
    main()
