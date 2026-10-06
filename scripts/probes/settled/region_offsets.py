"""region_offsets.py FRAMES_NPZ BOX OUT_PNG LABEL=DISCS_NPZ [...] — where a displayed surface stands off the target mesh in
one region of the display camera's picture: the discs kept by exterior_offset_probe.py (points, signed offset s from the
mesh in pitches, its Gaussian mean g1 over a pitch) projected as the studio camera of morph_4k.py / still_render.py sees
them (FRAMES_NPZ's target fixes the camera: azimuth 35, elevation 18, 3840 x 2160), each disc a dot of the lattice's
size, the nearest disc per pixel kept. BOX = x0,y0,x1,y1 in that picture. One panel per disc file over the box, coloured
by s (blue: inside the mesh, red: outside, +-1 pitch); printed per file over the discs seen in the box: the mean offset,
its rms, the fine part's rms (s - g1, below about 4 pitches), and the shares farther than half a pitch in and out."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402,F401  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap         # noqa: E402
from PIL import Image, ImageDraw, ImageFont                    # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center, radius = tgt.mean(0), float((tgt - tgt.mean(0)).norm(dim=1).max())
box = [int(v) for v in sys.argv[2].split(",")]
out = sys.argv[3]
W, H = 3840, 2160
az, el = math.radians(35.), math.radians(18.)
toward = center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
eye = center + 3.6 * radius * toward
view = world_to_view_torch(eye, center)
tan_y = math.tan(math.radians(15)); tan_x = tan_y * W / H
div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f29b98", "#b8302f"])


def picture(P, s):
    q = (view[:3, :3] @ P.T).T + view[:3, 3]
    u = ((q[:, 0] / (q[:, 2] * tan_x) + 1) * W) / 2 - .5
    v = ((1 - q[:, 1] / (q[:, 2] * tan_y)) * H) / 2 - .5
    depth = q[:, 2]
    r = 4                                                       # a disc's dot: about the lattice's size at this distance
    best = torch.full((H * W,), float("inf"), device=dev)
    who = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            x, y = (u + dx).round().long(), (v + dy).round().long()
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            pix = y[ok] * W + x[ok]
            best.scatter_reduce_(0, pix, depth[ok], "amin")
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            x, y = (u + dx).round().long(), (v + dy).round().long()
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            idx = torch.nonzero(ok).squeeze(1)
            pix = y[ok] * W + x[ok]
            hit = depth[ok] <= best[pix] + 1e-6
            who[pix[hit]] = idx[hit]
    return who.view(H, W)


font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
bw, bh = box[2] - box[0], box[3] - box[1]
specs = [a.split("=", 1) for a in sys.argv[4:]]
sheet = Image.new("RGB", (len(specs) * bw, bh + 46), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for i, (label, path) in enumerate(specs):
    Z = np.load(path)
    P = torch.as_tensor(Z["points"], device=dev)
    s = torch.as_tensor(Z["s"], device=dev)
    g1 = torch.as_tensor(Z["g1"], device=dev)
    who = picture(P, s)[box[1]:box[3], box[0]:box[2]]
    seen = torch.unique(who[who >= 0])
    ss, ff = s[seen], (s - g1)[seen]
    print(f"{label}: {len(seen)} discs seen in the box; mean offset {float(ss.mean()):+.3f}, rms {float(ss.square().mean().sqrt()):.3f}, "
          f"fine rms {float(ff.square().mean().sqrt()):.3f}; out > 0.5: {float((ss > .5).float().mean()):.3f}, in < -0.5: "
          f"{float((ss < -.5).float().mean()):.3f}", flush=True)
    col = np.full((bh, bw, 3), 255, np.uint8)
    w_np = who.cpu().numpy()
    m = w_np >= 0
    col[m] = (np.asarray(div(np.clip((s[torch.as_tensor(w_np[m], device=dev)].cpu().numpy() + 1) / 2, 0, 1)))[:, :3] * 255).astype(np.uint8)
    sheet.paste(Image.fromarray(col), (i * bw, 46))
    dr.text((i * bw + 10, 6), label, fill=(20, 20, 20), font=font)
sheet.save(out)
print("wrote", out, "(blue: inside the mesh, red: outside, +-1 pitch)", flush=True)
