"""box_probe.py ARCHIVE_NPZ RUN_JSON FRAMES_DIR OUT_DIR RAW:x0,y0,x1,y1 [RAW:x0,y0,x1,y1 ...] — what the particles drawn
in a 4K box of one frame are, and where they end.

With the 4K renderer's camera (azimuth 35, elevation 18, fov 30, distance 3.6 radii) and its splat rule (disc sigma =
target spacing x clamp(8th-neighbour distance / coverage radius, 1, 4); opacity by live support), for the rendered
particles that project into the box at raw frame RAW:
  attached and dense      connected to the body (single linkage at one layer spacing), disc not inflated (< 1.5);
  attached and stretched  connected, disc inflated 1.5x or more: the surface drawn with large soft discs;
  detached                not connected to the body: single particles and groups, with their disc inflation.
Per class: how many, the disc inflation (median, 90th percentile), the distance to the body (detached: gap in layer
spacings), and the fate at the end of the run: the share within the sampling berth of the target, the share detached
there, the median distance to the target. An overlay of the box (FRAMES_DIR/<RAW / stride>.png, stride from the frame
count) marks detached particles red and attached stretched ones yellow, beside the plain crop."""
import glob, json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.pipeline.window.layer import detached_groups, layer_spacing  # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
frames_dir, out_dir = sys.argv[3], sys.argv[4]
os.makedirs(out_dir, exist_ok=True)
cfg = run["arms"]["render_full_dt_iso_nn"]["config"]
frames, n_del = z["frames"], int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
sp, cov = float(td[:, 1].median()), float(td[:, 8].median())
berth = float(cfg.get("nn_berth_k", 1.97)) * sp
tknn = gpu.KNN(tgt)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
az, el = math.radians(AZ), math.radians(EL)
cam = center + 3.6 * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(cam, center)
tan_y = math.tan(math.radians(30) / 2); tan_x = tan_y * W / H
pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
stride = max(1, round((n_del - 1) / max(len(pngs) - 1, 1)))
X = lambda raw: torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)  # noqa: E731


def state(x):
    d, nb = knn_self_torch(x, 33)
    spacing = layer_spacing(x)
    group, size = detached_groups(d, nb, spacing)
    return d, group, size, spacing


xe = X(n_del - 1)
d_e, group_e, _, _ = state(xe)
dist_e = tknn.query(xe, 1)[0][:, 0].float()
print(f"N {len(xe)}, {n_del} delivered frames, render stride {stride}; target spacing sp {sp:.4f} wu, berth {berth / sp:.2f} sp")
for arg in sys.argv[5:]:
    raw, b = arg.split(":")
    raw, (x0, y0, x1, y1) = int(raw), [int(v) for v in b.split(",")]
    x = X(raw)
    d, group, size, spacing = state(x)
    p = x @ view[:3, :3].T + view[:3, 3]
    u = (p[:, 0] / (p[:, 2] * tan_x) + 1) / 2 * W
    v = (1 - p[:, 1] / (p[:, 2] * tan_y)) / 2 * H
    infl = (d[:, 8] / cov).clamp(1, 4)
    sup = live_support(d, cov, sp)
    inbox = (u >= x0) & (u < x1) & (v >= y0) & (v < y1) & (sup > 0)
    det = group > 0
    body = x[~det]
    classes = (("attached and dense    ", inbox & ~det & (infl < 1.5)), ("attached and stretched", inbox & ~det & (infl >= 1.5)),
               ("detached, single      ", inbox & det & (size == 1)), ("detached, in a group  ", inbox & det & (size > 1)))
    tot_all = int(det.sum())
    print(f"\n== raw {raw} (window {raw / (2 * int(cfg['T'])):.1f}), box {x0},{y0},{x1},{y1}: {int(inbox.sum())} rendered particles in the box; layer spacing {spacing / sp:.2f} sp; "
          f"in the whole frame {tot_all} detached particles ({int((det & (sup > 0)).sum())} rendered)")
    print("   class | particles | disc inflation median, p90 | gap to the body, layer spacings (median) | at the end: within the berth % | detached % | distance to the target sp (median)")
    for name, m in classes:
        c = int(m.sum())
        if c == 0:
            print(f"   {name} | 0")
            continue
        ids = torch.nonzero(m).squeeze(1)
        gap = "-"
        if "detached" in name:
            sub = ids[torch.randperm(len(ids), device=dev)[:2000]]
            gap = f"{float(torch.cdist(x[sub], body).min(1).values.median()) / spacing:.1f}"
        print(f"   {name} | {c:6d} | {float(infl[ids].median()):.2f}, {float(infl[ids].quantile(.9)):.2f} | {gap} | {100 * float((dist_e[ids] <= berth).float().mean()):3.0f} | "
              f"{100 * float((group_e[ids] > 0).float().mean()):3.0f} | {float(dist_e[ids].median()) / sp:.1f}")
    png = pngs[min(round(raw / stride), len(pngs) - 1)]
    crop = Image.open(png).convert("RGB").crop((x0, y0, x1, y1))
    over = crop.copy()
    dr = ImageDraw.Draw(over)
    for name, m, col in (("stretched", inbox & ~det & (infl >= 1.5), (255, 220, 0)), ("detached", inbox & det, (255, 40, 40))):
        ids = torch.nonzero(m).squeeze(1)
        for a_, b_ in zip((u[ids] - x0).tolist(), (v[ids] - y0).tolist()):
            dr.point((a_, b_), fill=col)
    sheet = Image.new("RGB", (2 * crop.width + 6, crop.height), (20, 20, 20))
    sheet.paste(crop, (0, 0)); sheet.paste(over, (crop.width + 6, 0))
    name = os.path.join(out_dir, f"box_{raw}_{x0}_{y0}.jpg")
    sheet.save(name, quality=92)
    print(f"   wrote {name} ({os.path.basename(png)}; red = detached, yellow = attached with a disc inflated 1.5x or more)")
