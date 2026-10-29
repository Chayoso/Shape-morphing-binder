"""rgv_crevice.py OUT_DIR VIZ_NPZ DUMP_DIR RUN_JSON W0 W1 -- why panel C hatches the dragon's hind-leg crevice and flank
(the render's share under 5 %) at 3-6 s: per committed window W0..W1 (1-based) of the RGV run, on the window's start
state (the dump's x0 and outer layer), CPU only.

Regions (outer-layer particles):
  crevice  D148's hind leg-body crevice box on the deliverable's 4K side view (az 90, el 18; pixels 1550-1950 x
           1150-1550, d148_cleanup_pull.py's camera), all depths;
  flank    the box above it on the same view (1250-2150 x 750-1150), all depths;
  horns    the top 3 % of the layer in height; toes: the bottom 3 %.
Per region: the render's share s = |lambda g_r| / (|lambda g_r| + |g_p|) (the panel's quantity), |lambda g_r| and
|g_p| (the window's first gradient on dF_c, summed over the driven steps), the render gradient's silhouette / shading
split (the dump's gl_sil / gl_pbr, squared-norm shares), the cleanup's part of the physics gradient (gl_dt) when
dumped. Visibility in the render loss's 18 views (6 azimuths x elevations 0, +-0.5; orthographic, 188 px, half-extent
1.25 max |target|, as physmorph.losses.silhouette): every layer particle splatted to its pixel, a z-buffer along the
view's camera axis (right x up); a particle is seen when within 1.5 pitches of the pixel's front, at the silhouette
edge when seen and within one pixel of an edge pixel (occupied, with an empty 4-neighbour) of the whole body's
occupancy. Writes OUT_DIR/crevice.json and OUT_DIR/views_wNN.png (the 18 views at 188 px, x2, depth-shaded, the
crevice's seen particles orange, its hidden ones' pixels dark red, the flank cyan)."""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

out, viz_p, dump_dir, run_json, W0, W1 = sys.argv[1:7]
W0, W1 = int(W0), int(W1)
os.makedirs(out, exist_ok=True)
z = np.load(viz_p)
tgt = np.asarray(z["tgt"], np.float64)
pitch = float(z["pitch"])
lam = np.asarray(z["w_lam"])
anim = np.asarray(z["w_anim"]).astype(int)
t0, t1 = np.asarray(z["w_t0"]), np.asarray(z["w_t1"])
R, P = z["g_rend"], z["g_phys"]
extent = float(np.abs(tgt).max()) * 1.25
RES = 188
views = [(float(t + (i / 3) * (2 * np.pi / 6)), float(ph)) for i, ph in enumerate((0.0, 0.5, -0.5))
         for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]


def side_pixels(p):
    """d148_cleanup_pull.camera: the deliverable's 4K side view."""
    center = tgt.mean(0)
    radius = float(np.linalg.norm(tgt - center, axis=1).max())
    W, H, AZ, EL = 3840, 2160, math.radians(90.0), math.radians(18.0)
    cam = center + 3.6 * radius * np.array([math.cos(EL) * math.sin(AZ), math.sin(EL), math.cos(EL) * math.cos(AZ)])
    fwd = (center - cam) / np.linalg.norm(center - cam)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    ty = math.tan(math.radians(15.0))
    tx = ty * W / H
    d = p - cam
    xc, yc, zc = d @ right, d @ up, d @ fwd
    return np.stack((((xc / zc) / tx + 1) / 2 * W - 0.5, (1 - (yc / zc) / ty) / 2 * H - 0.5), 1)


def in_box(q, b):
    return (q[:, 0] >= b[0]) & (q[:, 0] <= b[2]) & (q[:, 1] >= b[1]) & (q[:, 1] <= b[3])


CREVICE, FLANK = (1550, 1150, 1950, 1550), (1250, 750, 2150, 1150)
res = {"pitch": pitch, "extent": extent, "views": views, "boxes": {"crevice": CREVICE, "flank": FLANK}, "windows": {}}
for w in range(W0 - 1, W1):
    d = np.load(os.path.join(dump_dir, f"win_{anim[w]:04d}.npz"))
    x0 = d["x0"].astype(np.float64)
    lay = d["layer_mask"] > 0.5
    q = side_pixels(x0)
    yl = x0[lay, 1]
    hi, lo = np.percentile(yl, 97), np.percentile(yl, 3)
    regions = {"crevice": lay & in_box(q, CREVICE), "flank": lay & in_box(q, FLANK) & ~in_box(q, CREVICE),
               "horns": lay & (x0[:, 1] >= hi), "toes": lay & (x0[:, 1] <= lo), "layer": lay}
    gr, gp = np.asarray(R[w], np.float64), np.asarray(P[w], np.float64)
    s = gr / np.maximum(gr + gp, 1e-300)
    lr = float(d["lam_r"])
    gsil = lr * d["gl_sil_psum"].astype(np.float64) if "gl_sil_psum" in d.files else None
    gpbr = lr * d["gl_pbr_psum"].astype(np.float64) if "gl_pbr_psum" in d.files else None
    gdt = d["gl_dt_psum"].astype(np.float64) if "gl_dt_psum" in d.files else None
    # the 18 views: orthographic z-buffer of the layer, the body's occupancy and its silhouette edge
    th = np.array([v[0] for v in views])
    ph = np.array([v[1] for v in views])
    right = np.stack([np.cos(th), np.zeros_like(th), -np.sin(th)], 1)
    up = np.stack([-np.sin(ph) * np.sin(th), np.cos(ph), -np.sin(ph) * np.cos(th)], 1)
    toward = np.cross(right, up)
    seen = np.zeros((len(views), len(x0)), bool)
    edge_hit = np.zeros((len(views), len(x0)), bool)
    tiles = []
    for v in range(len(views)):
        uv = np.stack([x0 @ right[v], x0 @ up[v]], 1)
        pix = np.floor((uv + extent) / (2 * extent) * RES).astype(int)
        ok = (pix >= 0).all(1) & (pix < RES).all(1)
        zz = x0 @ toward[v]
        occ = np.zeros((RES, RES), bool)
        occ[pix[ok, 0], pix[ok, 1]] = True
        zbuf = np.full((RES, RES), -np.inf)
        li = np.where(ok)[0]                          # every particle: the front is the layer, no holes between
        np.maximum.at(zbuf, (pix[li, 0], pix[li, 1]), zz[li])
        sv = ok & lay & (zz >= zbuf[np.clip(pix[:, 0], 0, RES - 1), np.clip(pix[:, 1], 0, RES - 1)] - 1.5 * pitch)
        seen[v] = sv
        pad = np.pad(occ, 1)
        edge = occ & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        near = edge.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                near |= np.roll(np.roll(edge, di, 0), dj, 1)
        edge_hit[v] = sv & near[np.clip(pix[:, 0], 0, RES - 1), np.clip(pix[:, 1], 0, RES - 1)]
        # the tile: depth-shaded body, the crevice (seen orange, hidden dark red), the flank cyan
        zn = np.where(np.isfinite(zbuf), zbuf, np.nan)
        lo_z, hi_z = np.nanmin(zn), np.nanmax(zn)
        img = np.full((RES, RES, 3), (60, 66, 74), np.uint8)
        shade = np.clip(0.35 + 0.65 * (zn - lo_z) / max(hi_z - lo_z, 1e-9), 0, 1)
        body = np.isfinite(zbuf)
        img[body] = (np.stack([shade[body]] * 3, 1) * np.array([205, 210, 214])).astype(np.uint8)
        for m, col in ((regions["flank"] & sv, (60, 200, 220)), (regions["crevice"] & ok & ~sv, (130, 20, 20)),
                       (regions["crevice"] & sv, (255, 140, 30))):
            img[pix[m, 0], pix[m, 1]] = col
        img[edge] = (255, 255, 255)                   # the silhouette edge: where the silhouette term acts
        img = np.flip(np.transpose(img, (1, 0, 2)), 0)           # u to the right, v up
        tiles.append(Image.fromarray(img).resize((2 * RES, 2 * RES), Image.NEAREST))
    sheet = Image.new("RGB", (6 * 2 * RES + 7 * 6, 3 * (2 * RES + 22) + 30), (40, 44, 50))
    dr = ImageDraw.Draw(sheet)
    dr.text((8, 6), f"dragon 300k RG97, window {w + 1} ({t0[w]:.2f}-{t1[w]:.2f} s): the render loss's 18 views at {RES} px "
            f"(x2); orange = the crevice's particles the view sees, dark red = hidden, cyan = the flank seen, white = the silhouette's edge (where the silhouette term pulls)", fill=(235, 235, 235))
    for v, tile in enumerate(tiles):
        r_, c_ = divmod(v, 6)
        x_, y_ = 6 + c_ * (2 * RES + 6), 30 + r_ * (2 * RES + 22)
        sheet.paste(tile, (x_, y_ + 16))
        nc = int((regions["crevice"] & seen[v]).sum())
        dr.text((x_, y_), f"az {math.degrees(views[v][0]):.0f}, el {math.degrees(views[v][1]):+.0f}: crevice seen {nc}",
                fill=(235, 235, 235))
    sheet.save(os.path.join(out, f"views_w{w + 1:03d}.png"))
    row = {"t": [float(t0[w]), float(t1[w])], "lambda": lr}
    for name, m in regions.items():
        if not m.any():
            continue
        r = {"n": int(m.sum()), "share_median": float(np.median(s[m])), "render_median": float(np.median(gr[m])),
             "physics_median": float(np.median(gp[m])),
             "views_seen_median": float(np.median(seen[:, m].sum(0))),
             "views_seeing_10pct": int(((seen[:, m].mean(1)) >= 0.1).sum()),
             "seen_pairs_at_edge": float(edge_hit[:, m].sum() / max(seen[:, m].sum(), 1)),
             "never_seen": float((seen[:, m].sum(0) == 0).mean())}
        if gsil is not None and gpbr is not None:
            ss, sp = (gsil[m] ** 2).sum(), (gpbr[m] ** 2).sum()
            r["render_sil_share"] = float(ss / max(ss + sp, 1e-300))
        if gdt is not None:
            r["cleanup_over_physics_median"] = float(np.median(gdt[m] / np.maximum(gp[m], 1e-300)))
        row[name] = r
    res["windows"][w + 1] = row
    print(w + 1, {k: (v["n"], round(v["share_median"], 3), round(v["views_seen_median"], 1), v["views_seeing_10pct"],
                      round(v["seen_pairs_at_edge"], 3), round(v.get("render_sil_share", -1), 2))
                  for k, v in row.items() if isinstance(v, dict)}, flush=True)
json.dump(res, open(os.path.join(out, "crevice.json"), "w"), indent=1)
print("written", out, flush=True)
