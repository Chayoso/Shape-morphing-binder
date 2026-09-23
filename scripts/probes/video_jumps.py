"""Per-frame discontinuities of a morph video (2026-09-23): for every video frame the image change
against the previous frame (mean |dI| on a 480x240 grey downscale), the object's pixel area (the
per-pixel median over all frames is the static background; object = |I - bg| > 12/255) and its
change, and — from the QA sidecar and the archive — the re-mesh flag, the drawn / dropped component
counts and the particles' mean displacement of that archived frame. A "wiped connection" is a
frame where the area drops and the image jumps while the particles do not move more than usual;
the sidecar says whether the surface was re-meshed there.

usage: video_jumps.py <mp4> [--sidecar f.components.txt] [--npz archive.npz] [--stride 3] [--top 8] [--out series.txt]
"""
import subprocess
import sys

import numpy as np


def opt(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def read_video_gray(path, w=480, h=240):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[: n * w * h], np.uint8).reshape(n, h, w).astype(np.float32) / 255.0


def main():
    mp4 = sys.argv[1]
    sidecar = opt("--sidecar", None)
    npz = opt("--npz", None)
    stride = opt("--stride", 3, int)
    top = opt("--top", 8, int)
    out = opt("--out", None)
    V = read_video_gray(mp4)
    n = len(V)
    bg = np.median(V, axis=0)
    mask = np.abs(V - bg) > 12.0 / 255.0
    area = mask.reshape(n, -1).mean(1)
    darea = np.zeros(n); darea[1:] = (area[1:] - area[:-1]) / np.maximum(area[:-1], 1e-6)
    dI = np.zeros(n); dI[1:] = np.abs(V[1:] - V[:-1]).reshape(n - 1, -1).mean(1)
    # the change concentrated in a region: the largest 2 % of pixel changes carry what share of dI
    conc = np.zeros(n)
    for k in range(1, n):
        d = np.abs(V[k] - V[k - 1]).ravel()
        q = np.quantile(d, 0.98)
        conc[k] = d[d >= q].sum() / max(d.sum(), 1e-9)
    remesh = np.zeros(n, int); comps = np.zeros(n, int); dropped = np.zeros(n, int); drift = np.full(n, np.nan); jit = np.full(n, np.nan)
    if sidecar:
        for line in open(sidecar):
            if line.startswith("#") or line.startswith("archived_frame"):
                continue
            p = line.split()
            if len(p) < 9:
                continue
            k = int(p[0]) // stride
            if k < n:
                comps[k] = int(p[1]); dropped[k] = int(p[3]); remesh[k] = int(p[8])
                jit[k] = float(p[6]) if p[6] != "nan" else np.nan; drift[k] = float(p[7]) if p[7] != "nan" else np.nan
    pmove = np.full(n, np.nan)
    if npz:
        z = np.load(npz); fr = z["frames"]
        for k in range(1, n):
            a, b = k * stride, (k - 1) * stride
            if a < len(fr):
                pmove[k] = float(np.linalg.norm(np.asarray(fr[a], np.float32) - np.asarray(fr[b], np.float32), axis=1).mean())
    med = np.median(dI[1:]) if n > 1 else 0.0
    order = np.argsort(-dI)[:top]
    print(f"{mp4.split('/')[-1]}: {n} frames, median |dI| {med:.4f}, area {area.mean() * 100:.1f} % of the frame, re-meshes {remesh.sum()}")
    print("  top image jumps: frame(archived) |dI|/median  area change  conc98  remeshed  components/dropped  particle move(wu)")
    for k in order:
        print(f"    {k:4d} ({k * stride:5d})  {dI[k] / max(med, 1e-9):5.1f}x   {darea[k] * 100:+6.1f} %   {conc[k]:.2f}     {remesh[k]}       {comps[k]}/{dropped[k]}            {pmove[k]:.4f}")
    big = [k for k in range(1, n) if darea[k] < -0.03]
    print(f"  frames with area drop > 3 %: {len(big)} -> {[(k, k * stride, round(darea[k] * 100, 1), int(remesh[k])) for k in big[:12]]}")
    r = np.where(remesh[1:] > 0)[0] + 1
    if len(r):
        print(f"  at re-mesh frames: mean |dI| {dI[r].mean() / max(med, 1e-9):.1f}x median, mean |area change| {np.abs(darea[r]).mean() * 100:.2f} %, max area drop {darea[r].min() * 100:+.1f} %; elsewhere mean |area change| {np.abs(np.delete(darea[1:], r - 1)).mean() * 100:.2f} %")
    if out:
        np.savetxt(out, np.column_stack([np.arange(n), np.arange(n) * stride, dI, area, darea, conc, remesh, comps, dropped, jit, drift, pmove]), fmt="%.5g",
                   header="video_frame archived_frame dI area darea conc98 remeshed components dropped refit_jitter track_drift particle_move")


if __name__ == "__main__":
    main()
