"""demo_compute.py MESH VIEWS OUT_NPZ [time] -- D157 demo (2026-10-21): what the render overlays and heatmaps draw.

Measurement only, CPU. 300k, seed 97, B1's render arm L and its lambda = 0 twin P. Constants as D157's reader: pitch and
box from the run's own target sample; the referee the exact mesh in the run's frame (open meshes fan-closed), raw and
grown by c (the perfect sample's mean exterior offset, D157's own reading of that mesh-seed).
Default: the end states. Per chosen held-out view, 1024-px orthographic silhouettes of the exterior (Zhu-Bridson zero
set) of L and P and of both referees; and a studio-camera (perspective, 3.6 radii, 30 deg) cast of the display surface
(zbs) of L and P at 1024 px: per pixel the signed distance of the visible surface to the exact mesh (+ outside, pitches)
and a headlight shade.
`time`: the kept frames at t = 0.5 / 1 / 2 / 4 s (raw frames 120 / 240 / 480 / 960) of L and P, their exterior's
silhouette in the first chosen view only."""
import glob
import json
import sys

import numpy as np

import d157_read as R

mesh, views, out = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3]
time_mode = len(sys.argv) > 4 and sys.argv[4] == "time"
G = R.G
arch = {lab: f"{G}/d145t_b1/{mesh}300k_{lab}97_end1.npz" for lab in ("L", "P")}
tgt = np.asarray(np.load(arch["L"])["tgt"], np.float32)
a = .708 * R.median_kth_spacing(tgt, 8, 20000)
ext = float(np.abs(tgt).max()) * 1.25
c = None
for f in glob.glob(f"{G}/d157/read300_*.jsonl"):
    for ln in open(f):
        r = json.loads(ln)
        if r["mesh"] == mesh and r["seed"] == 97 and r["label"] == "own":
            c = r["ctx"]["c_a"] * r["ctx"]["pitch"]
            assert abs(r["ctx"]["pitch"] - a) < 1e-9 * a + 1e-12, (r["ctx"]["pitch"], a)
m = np.load(f"{R.MESHDIR['300k']}/{mesh}_97.npz")
t, _ = R.close_holes(np.asarray(m["V"], np.float64), np.asarray(m["F"], np.int64))
if t.volume < 0:
    t.invert()
raw = R.Surf(np.asarray(t.vertices), np.asarray(t.faces))
V = [R.HELD[i] for i in views]
save = dict(pitch=a, extent=ext, c=c, views=np.array(views))


def masks(surf, vs):
    return np.stack([surf.occluded(R.ortho_rays(v, ext, 1024)).reshape(1024, 1024) for v in vs])


if time_mode:
    for lab in ("L", "P"):
        F = f"{G}/d145t_b1/{mesh}300k_{lab}97_frames12.npz"
        raws = [int(x) for x in np.load(F)["raws"]]
        for rw in (120, 240, 480, 960):
            i = raws.index(rw)
            x = _frame = None
            import zipfile
            from numpy.lib import format as npf
            with zipfile.ZipFile(F) as zf, zf.open("frames.npy") as fh:
                ver = npf.read_magic(fh)
                shape, fortran, dtype = npf._read_array_header(fh, ver)
                row = int(np.prod(shape[1:])) * dtype.itemsize
                fh.seek(fh.tell() + i * row)
                x = np.frombuffer(fh.read(row), dtype).reshape(shape[1:]).astype(np.float32)
            st, _ = R.field_mesh("zb", x, a)
            save[f"{lab}_t{rw}"] = np.packbits(masks(st, V[:1]))
            print(mesh, lab, rw, flush=True)
    np.savez_compressed(out, **save)
    sys.exit(0)

grown, gi = R.grown_mesh(raw, c, a)
save["raw_mask"] = np.packbits(masks(raw, V))
save["cor_mask"] = np.packbits(masks(grown, V))
center = tgt.astype(np.float64).mean(0)
radius = float(np.linalg.norm(tgt - center, axis=1).max())
for lab in ("L", "P"):
    x = np.asarray(np.load(arch[lab])["frames"][-1], np.float32)
    st, _ = R.field_mesh("zb", x, a)
    save[f"{lab}_mask"] = np.packbits(masks(st, V))
    disp, _ = R.field_mesh("zbs", x, a)
    dmaps, shades = [], []
    for v in V:
        cam, dirs, right, up, fwd = R.studio_cam(center, radius, v, 1024)
        rays = np.concatenate([np.broadcast_to(cam, dirs.shape).reshape(-1, 3), dirs.reshape(-1, 3)], 1)
        hit, pid, uv, th = disp.cast(rays)
        d = np.full(len(rays), np.nan)
        P = rays[hit, :3] + th[hit, None] * rays[hit, 3:]
        d[hit] = raw.signed(P) / a
        n = np.zeros((len(rays), 3))
        n[hit] = disp.normals_at(pid[hit], uv[hit])
        sh = np.clip((n * -rays[:, 3:]).sum(1), 0, 1)
        dmaps.append(d.reshape(1024, 1024).astype(np.float16))
        shades.append(sh.reshape(1024, 1024).astype(np.float16))
    save[f"{lab}_dist"] = np.stack(dmaps)
    save[f"{lab}_shade"] = np.stack(shades)
    print(mesh, lab, "done", flush=True)
np.savez_compressed(out, **save)
print("saved", out, "c/a", c / a, "grown", gi, flush=True)
