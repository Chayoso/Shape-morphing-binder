"""d173_studio.py (D173: d170_studio.py + +colour=nn) -- was d170_studio.py (D170: d160_studio.py with each tile flipped back left-right and the columns twin | full | paint) -- was d160_studio.py OUT_DIR SCENE FULL_FRAMES12 TWIN_FRAMES12 IND_NPZ [+stride=1] [+w=960] [+h=540] [+endonly=1] -- D160's videos:
the two arms of one seed side by side in the deliverables' studio look (render/studio.StudioRaster: the grey gradient
backdrop, the three lights, the tone map; scripts/render_deliverable.py's base display, +fit=0: the exterior discs of
each kept frame's particles on the display's field (zbs, D144), their normals averaged twice over the reach, discs of a
lattice of 0.46 run pitches). The one change from the studio look: each disc's albedo is the studio's grey times its
particles' albedos (the kernel mean, render/exterior.found_disc_albedo), so white material is the studio grey and red
material red. Columns: "dF_c + layer u + render loss (full)", "dF_c + layer u (lambda = 0 twin)", "target paint" (the
independent sample painted by the scene, at rest); rows: the front (azimuth 35, the deliverables') and the back (215).
Writes OUT_DIR/frames/NNNN.png, OUT_DIR/side_by_side.mp4 (20 fps, the kept frames' real time) and OUT_DIR/end.png."""
import subprocess
import sys
from pathlib import Path

from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf
from PIL import Image, ImageDraw, ImageFont

from physmorph.pipeline.appearance import RED, scene_params, source_red, target_red
from physmorph.render.exterior import Lattice, found_disc_albedo, make_field
from physmorph.render.knn_gpu import knn_self_torch
from physmorph.render.studio import StudioRaster
from physmorph.render.support import normal_filter_size

dev = torch.device("cuda")
opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv if not s.startswith("+")]
out, scene = Path(argv[1]), argv[2]
zf, zt = (np.load(p, allow_pickle=True) for p in argv[3:5])
ind = np.asarray(np.load(argv[5])["tgt"], np.float32)
STRIDE, W, H = int(opts.get("stride", 1)), int(opts.get("w", 960)), int(opts.get("h", 540))
NN = opts.get("colour", "kernel") == "nn"   # D173: +colour=nn = each disc the colour of its nearest particle (the identity, no kernel)
END_ONLY = opts.get("endonly", "0") == "1"                     # +endonly=1: the delivered end only (the sheets)
(out / "frames").mkdir(parents=True, exist_ok=True)
tgt = torch.as_tensor(np.asarray(zf["tgt"], np.float32), device=dev)
pitch = lambda s: .708 * float(knn_self_torch(s, 9)[0][:, 8].median())   # noqa: E731
a = pitch(tgt)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
h_lat = .46 * a
lat = Lattice(center, 2.8 * radius)
STUDIO = (.64, .69, .72)                                       # StudioRaster's albedo (the studio grey)
cams = [StudioRaster(center, radius, W, H, az, 18.) for az in (35., 215.)]


def paint(red):
    alb = np.where(np.asarray(red)[:, None], np.asarray(RED, np.float32), np.ones(3, np.float32))
    return torch.as_tensor(alb, device=dev)


def surface(x, alb):
    """(points, averaged normals, display albedo) of a state's base display."""
    with torch.no_grad():
        f = make_field("zbs", x, a, device_field=True)
        p, g, _, _ = lat.discs(f, h_lat)
        n = nnf.normalize(g, dim=1)
        d, nbx = knn_self_torch(x, 33)
        reach = float(d[(x - x[nbx[:, 1:]].mean(1)).norm(dim=1) >= .5 * a / .708, 32].median())
        spacing = float(knn_self_torch(p, 2)[0][:, 1].median())
        dd, nn = gpu.KNN(p).query(p, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
        wr = (dd.float() <= reach)[..., None]
        for _ in range(2):
            n = nnf.normalize((n[nn] * wr).sum(1), dim=1, eps=1e-9)
        c = (alb[gpu.KNN(x).query(p, 1)[1][:, 0]] if NN else found_disc_albedo(f, p, alb, alb.mean(0), a)) * p.new_tensor(STUDIO)
    return p, n, c


def draw(cam, q, n, c):
    rr = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    tt = nnf.normalize(torch.linalg.cross(n, rr), dim=1, eps=1e-9)
    rot = torch.stack((tt, torch.linalg.cross(n, tt), n), dim=2)
    sig = torch.full((len(q),), h_lat, device=dev)              # the deliverable's disc: its lattice pitch
    cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
    op = torch.full((len(q),), .92, device=dev)
    with torch.no_grad():
        col = cam.raster_colors(q, n, cov, op, c)
        cv = cam.raster_colors(q, n, cov, op, torch.ones_like(c))[..., :1]
        cam.albedo = (col / cv.clamp_min(1e-4)).clamp(0, 1)
        img = cam(q, n, cov, op, normal_kernel=normal_filter_size(H, False))
        cam.albedo = q.new_tensor(STUDIO)
    return (img.clamp(0, 1) * 255).round().byte().cpu().numpy()


x_ind = torch.as_tensor(ind, device=dev)
if "src_albedo" in zf.files and "src_albedo" in zt.files:    # D173: the archives' own paint (any scene, the texture's too)
    alb_f = torch.as_tensor(np.asarray(zf["src_albedo"], np.float32), device=dev)
    alb_t = torch.as_tensor(np.asarray(zt["src_albedo"], np.float32), device=dev)
    zi = np.load(argv[5], allow_pickle=True)
    pi, ni, ci = surface(x_ind, torch.as_tensor(np.asarray(zi["tgt_albedo"], np.float32), device=dev))
else:
    prm = scene_params(scene, np.asarray(zf["frames"][0], np.float32), np.asarray(zf["tgt"], np.float32))
    alb_f = paint(source_red(prm, np.asarray(zf["frames"][0], np.float32)))
    alb_t = paint(source_red(prm, np.asarray(zt["frames"][0], np.float32)))
    pi, ni, ci = surface(x_ind, paint(target_red(prm, ind)))
tgt_imgs = [draw(cam, pi, ni, ci) for cam in cams]
try:
    font = ImageFont.load_default(size=max(14, H // 24))
except TypeError:
    font = ImageFont.load_default()
LABELS = ("dF_c + layer u (twin)", "dF_c + layer u + render loss (full)", "target paint")
nf, nt = len(zf["frames"]), len(zt["frames"])
n_frames = max(nf, nt)
keep = list(range(0, n_frames, STRIDE)) + ([n_frames - 1] if (n_frames - 1) % STRIDE else [])
if END_ONLY:
    keep = [n_frames - 1]
for i, k in enumerate(keep):
    tiles = []
    for z, alb, m in ((zt, alb_t, nt), (zf, alb_f, nf)):   # D170: twin | full | paint
        x = torch.as_tensor(np.asarray(z["frames"][min(k, m - 1)], np.float32), device=dev)
        q, n, c = surface(x, alb)
        tiles.append([draw(cam, q, n, c) for cam in cams])
    tiles.append(tgt_imgs)
    canvas = Image.new("RGB", (3 * W, 2 * H))
    for col, t in enumerate(tiles):
        for row in range(2):
            canvas.paste(Image.fromarray(np.ascontiguousarray(t[row][:, ::-1])), (col * W, row * H))   # D170: StudioRaster mirrors left-right; each tile flipped back
    dr = ImageDraw.Draw(canvas)
    for col, lab in enumerate(LABELS):
        dr.text((col * W + 12, 8), lab, fill=(235, 235, 235), font=font)
    t_s = float(np.asarray(zf["raws"])[min(k, nf - 1)]) / 240.0
    dr.text((12, 2 * H - H // 14), f"{scene}   t = {t_s:4.2f} s (full)", fill=(235, 235, 235), font=font)
    canvas.save(out / "frames" / f"{i:04d}.png")
    if k == keep[-1]:
        canvas.save(out / "end.png")
    print(i, k, flush=True)
if not END_ONLY:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", str(out / "frames" / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / "side_by_side.mp4")], check=True)
print("wrote", out, flush=True)
