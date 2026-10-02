"""frame_forensics.py NPZ RAW_FRAME[,RAW_FRAME...] OUT_DIR [RENDER_JSON FRAMES_DIR] — which particles make the artifact.

For each requested raw frame of a morph archive, with the 4K renderer's own camera and per-particle splat rule
(scripts/render_splat_photoreal.py: sigma = target spacing x clamp(8th-neighbour distance / target coverage radius,
1, 4); opacity 0.92 x live support), every particle gets: its pixel, depth and whether it is front-most; its splat
inflation; its support; its distance to the nearest target point. Printed and saved:
  floaters  = rendered particles (support > 0) farther than 3 target spacings from the target, by index, with their
              pixel, splat inflation and neighbour distance: the floating Gaussians of that frame;
  inflated  = front-most particles whose splat is inflated (>= 1.5x): where the image is drawn with large discs (blur);
  holes     = outer target points with no particle within 2 spacings, with their pixels.
An overlay PNG (1920 wide) marks them on the rendered frame when the render's JSON and frames directory are given:
red = floaters, yellow = inflated front particles, cyan = holes. The per-particle arrays are saved as NPZ so that a
particle index can be followed through other frames (trace mode: frame_forensics.py NPZ trace:ID,ID,... OUT_DIR)."""
import json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
path, what, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
z = np.load(path, allow_pickle=True)
frames = z["frames"]
n_del = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
td, _ = knn_self_torch(tgt, 9)
sp, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
tknn = gpu.KNN(tgt)
az, el = math.radians(AZ), math.radians(EL)
cam = center + 3.6 * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(cam, center)
tan_y = math.tan(math.radians(30) / 2); tan_x = tan_y * W / H
# the target's outer layer (the centroid of 33 neighbours is off the point)
dk, ik = knn_self_torch(tgt, 34)
outer_t = (tgt[ik[:, 1:]].mean(1) - tgt).norm(dim=1) > 0.35 * dk[:, -1]
ty = tgt[outer_t]
print(f"archive: {len(frames)} frames, {n_del} delivered, N {frames.shape[1]}; target spacing {sp:.4f}, coverage radius {cov_r:.4f} ({cov_r / sp:.2f} sp)")


def project(x):
    p = x @ view[:3, :3].T + view[:3, 3]
    zc = p[:, 2]
    px = (p[:, 0] / (zc * tan_x) + 1) / 2 * W
    py = (1 - p[:, 1] / (zc * tan_y)) / 2 * H
    return px, py, zc


def particle_state(raw):
    x = torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)
    d, _ = knn_self_torch(x, 33)
    support = live_support(d, cov_r, sp)
    infl = (d[:, 8] / cov_r).clamp(1., 4.)
    dt = tknn.query(x, 1)[0][:, 0] / sp
    px, py, zc = project(x)
    # front-most: within two spacings of the nearest depth in a 4-pixel cell
    cx, cy = (px / 4).long().clamp(0, W // 4 - 1), (py / 4).long().clamp(0, H // 4 - 1)
    cell = cy * (W // 4) + cx
    zmin = torch.full((W // 4 * (H // 4),), float("inf"), device=dev).scatter_reduce(0, cell, zc, "amin")
    front = (zc <= zmin[cell] + 2 * sp) & (support > 0)
    return x, d, support, infl, dt, px, py, zc, front


def overlay(raw, px, py, floaters, inflated, holes_px):
    if len(sys.argv) < 6:
        return None
    from PIL import Image, ImageDraw
    rj = json.loads(Path(sys.argv[4]).read_text())
    raws = rj.get("raw_frame_indices") or []
    if raw not in raws:
        raw_r = min(raws, key=lambda r: abs(r - raw))
        print(f"   raw frame {raw} was not rendered; the nearest rendered one is {raw_r}")
    else:
        raw_r = raw
    img = Image.open(Path(sys.argv[5]) / f"{raws.index(raw_r):04d}.png").convert("RGB")
    s = 1920 / img.width
    img = img.resize((1920, int(img.height * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(img)
    for i in inflated[:: max(1, len(inflated) // 4000)]:
        u, v = float(px[i]) * s, float(py[i]) * s
        dr.point((u, v), fill=(255, 220, 0))
    for u, v in holes_px[:: max(1, len(holes_px) // 2000)]:
        dr.ellipse((u * s - 1.5, v * s - 1.5, u * s + 1.5, v * s + 1.5), outline=(0, 255, 255))
    for i in floaters[:400]:
        u, v = float(px[i]) * s, float(py[i]) * s
        dr.ellipse((u - 6, v - 6, u + 6, v + 6), outline=(255, 40, 40), width=2)
    out = out_dir / f"overlay_{raw:05d}.png"
    img.save(out)
    return out


if what.startswith("trace:"):
    ids = torch.as_tensor([int(s) for s in what[6:].split(",")], device=dev)
    step = max(1, n_del // 80)
    print("raw frame | per particle: distance to target (sp) / 8th-neighbour distance over coverage radius / support")
    for raw in list(range(0, n_del, step)) + [n_del - 1]:
        x, d, support, infl, dt, px, py, zc, front = particle_state(raw)
        print(f"{raw:6d} | " + "  ".join(f"{int(i)}: {float(dt[i]):5.1f} / {float(d[i, 8] / cov_r):4.1f} / {float(support[i]):.2f}" for i in ids))
    sys.exit(0)

for raw in [int(s) for s in what.split(",")]:
    raw = min(raw, n_del - 1)
    x, d, support, infl, dt, px, py, zc, front = particle_state(raw)
    onscreen = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    fl = torch.nonzero((dt > 3) & (support > 0) & onscreen).squeeze(1)
    fl = fl[torch.argsort(dt[fl], descending=True)]
    inf = torch.nonzero(front & (infl >= 1.5)).squeeze(1)
    dmin = torch.cat([torch.cdist(ty[c:c + 4096], x).min(1).values for c in range(0, len(ty), 4096)]) / sp
    hole = dmin > 2
    hx, hy, _ = project(ty[hole])
    nf = int(front.sum())
    print(f"\n== raw frame {raw}: rendered particles (support > 0) {int((support > 0).sum())} of {len(x)}; front-most {nf}")
    print(f"   floaters (> 3 sp from the target, rendered): {len(fl)}; farthest {float(dt.max()):.1f} sp; beyond 1.5 sp: {int(((dt > 1.5) & (support > 0)).sum())}")
    print(f"   splat inflation over the front-most particles: median {float(infl[front].median()):.2f}, p90 {float(torch.quantile(infl[front], .9)):.2f}, "
          f"share >= 1.5x {100 * float((infl[front] >= 1.5).float().mean()):.1f} %, share at the 4x cap {100 * float((infl[front] >= 4).float().mean()):.2f} %")
    print(f"   support over all particles: zero {100 * float((support == 0).float().mean()):.2f} %, partial {100 * float(((support > 0) & (support < 1)).float().mean()):.2f} %")
    print(f"   holes: outer target points with no particle within 2 sp: {int(hole.sum())} of {len(ty)} ({100 * float(hole.float().mean()):.2f} %); within 1.5 sp uncovered {100 * float((dmin > 1.5).float().mean()):.2f} %")
    print("   the 15 farthest floaters: index | distance to target (sp) | 8NN / coverage | splat inflation | support | pixel (4K) | world xyz")
    for i in fl[:15]:
        i = int(i)
        print(f"     {i:7d} | {float(dt[i]):6.1f} | {float(d[i, 8] / cov_r):5.1f} | {float(infl[i]):.2f} | {float(support[i]):.2f} | ({float(px[i]):6.0f}, {float(py[i]):6.0f}) | "
              + " ".join(f"{float(v):+.3f}" for v in x[i]))
    np.savez_compressed(out_dir / f"particles_{raw:05d}.npz", px=px.cpu().numpy(), py=py.cpu().numpy(), depth=zc.cpu().numpy(),
                        front=front.cpu().numpy(), inflation=infl.cpu().numpy(), support=support.cpu().numpy(),
                        dist_target_sp=dt.cpu().numpy(), d8=d[:, 8].cpu().numpy(), floaters=fl.cpu().numpy())
    ov = overlay(raw, px, py, fl.tolist(), inf.tolist(), list(zip(hx.tolist(), hy.tolist())))
    if ov:
        print(f"   overlay: {ov}")
