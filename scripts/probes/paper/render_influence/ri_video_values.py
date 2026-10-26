"""ri_video_values.py MESH OUT_DIR -- render influence video, pass 1 (CPU; measurement only): on every kept frame of the
full arm L97 ("dF_c + layer u + render loss (full)", tag freeze-2026-10-15, 300k, seed 97), at its near-surface particles
(the pipeline's outer layer by neighbour asymmetry, and every particle whose exterior field is above -0.5 pitch), the displacement of its surface from the twin's at the same simulated time:
    d_i = (f_P97(x_i) - f_L97(x_i)) / a     (+ : the full arm's surface lies further OUT than the twin's here)
and the run-to-run noise of the same arm at the same seed (RG97, D145 B7):
    e_i = (f_RG97(x_i) - f_L97(x_i)) / a,
f the exterior field the loss reads (Zhu-Bridson, radius 3, offset 0.8 a). Each is then averaged over the surface
around the particle (Gaussian over its 32 nearest near-surface particles, sigma 1.5 pitches; the noise's square over
its 64 nearest, sigma 2 pitches). Where the twin's or the re-run's archive has ended (the stop rule ends runs at
different times) its last kept frame is used (the delivered end, at rest). Writes OUT_DIR/vals_MESH_{d,e2}.npy (float16,
kept frames x particles, NaN off the surface band) and OUT_DIR/vals_MESH.json (times, matches, per-frame rms)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.render.exterior import ZhuBridson
from physmorph.pipeline.window.layer import layer_by_asymmetry

torch.set_grad_enabled(False)
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"


def pitch(tgt):
    sub = tgt[np.random.default_rng(0).choice(len(tgt), 20000, replace=False)]
    d = cKDTree(sub).query(sub, 9)[0][:, 8]
    return .708 * float(np.median(d)) * (20000 / len(tgt)) ** (1 / 3)


def spacing8(x):
    xn = x.numpy()
    sub = xn[np.random.default_rng(0).choice(len(xn), 20000, replace=False)]
    return float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (20000 / len(xn)) ** (1 / 3)


def smooth(p, v, k, sig):
    d, nb = cKDTree(p).query(p, k, workers=8)
    w = np.exp(-0.5 * (d / sig) ** 2)
    return (w * v[nb]).sum(1) / w.sum(1)


def main():
    mesh, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    zL = np.load(f"{G}/d145t_b1/{mesh}300k_L97_frames12.npz", allow_pickle=True)
    FL, rawL = zL["frames"], np.asarray(zL["raws"]).astype(int)
    a = pitch(np.asarray(zL["tgt"], np.float64))
    zP = np.load(f"{G}/d145t_b1/{mesh}300k_P97_frames12.npz", allow_pickle=True)
    FP, rawP = zP["frames"], np.asarray(zP["raws"]).astype(int)
    zR = np.load(f"{G}/d145t_b7/{mesh}300k_RG97_frames12.npz", allow_pickle=True)
    FR, rawR = zR["frames"], np.asarray(zR["raws"]).astype(int)
    K, N = len(rawL), FL.shape[1]
    print(f"{mesh}: pitch {a:.5f}; kept frames L {K} (to {rawL[-1] / 240:.2f} s), P {len(rawP)} (to {rawP[-1] / 240:.2f} s), "
          f"R {len(rawR)} (to {rawR[-1] / 240:.2f} s); load {time.time() - t0:.0f} s", flush=True)
    D = np.lib.format.open_memmap(out / f"vals_{mesh}_d.npy", mode="w+", dtype=np.float16, shape=(K, N))
    E = np.lib.format.open_memmap(out / f"vals_{mesh}_e2.npy", mode="w+", dtype=np.float16, shape=(K, N))
    rows = []
    for k in range(K):
        r = rawL[k]
        kp = int(np.searchsorted(rawP, r)) if r <= rawP[-1] else len(rawP) - 1
        kr = int(np.searchsorted(rawR, r)) if r <= rawR[-1] else len(rawR) - 1
        xL = torch.as_tensor(np.asarray(FL[k], np.float32))
        fL = ZhuBridson(xL, a)(xL, grad=False)[0].numpy().astype(np.float64)
        lay = layer_by_asymmetry(xL, spacing8(xL))[0].numpy()     # the pipeline's outer layer (D159's R2 reads it)
        S = np.isfinite(fL) & (lay | (fL > -0.5 * a))           # the field is -0.8 a deep inside: only the shell counts
        q = xL[torch.as_tensor(S)]
        fP = ZhuBridson(torch.as_tensor(np.asarray(FP[kp], np.float32)), a)(q, grad=False)[0].numpy().astype(np.float64)
        fR = ZhuBridson(torch.as_tensor(np.asarray(FR[kr], np.float32)), a)(q, grad=False)[0].numpy().astype(np.float64)
        ok = np.isfinite(fP) & np.isfinite(fR)
        d = np.where(ok, (fP - fL[S]) / a, 0.0)
        e = np.where(ok, (fR - fL[S]) / a, 0.0)
        ps = q.numpy().astype(np.float64)
        ds = smooth(ps, d, 32, 1.5 * a)
        e2s = smooth(ps, e * e, 64, 2.0 * a)
        row = np.full(N, np.nan, np.float32)
        row[S] = ds
        D[k] = row.astype(np.float16)
        row[:] = np.nan
        row[S] = e2s
        E[k] = row.astype(np.float16)
        outer = lay[S]                                            # the outer layer, for the inset
        rows.append(dict(k=k, raw=int(r), t=r / 240.0, kp=kp, rawP=int(rawP[kp]), kr=kr, rawR=int(rawR[kr]),
                         n_surf=int(S.sum()), valid=float(ok.mean()),
                         rms_d=float(np.sqrt((d[outer] ** 2).mean())), rms_e=float(np.sqrt((e[outer] ** 2).mean())),
                         rms_ds=float(np.sqrt((ds[outer] ** 2).mean())), rms_es=float(np.sqrt(e2s[outer].mean())),
                         p99_ds=float(np.percentile(np.abs(ds[outer]), 99))))
        if k % 20 == 0 or k == K - 1:
            print(json.dumps(rows[-1]), f"{time.time() - t0:.0f} s", flush=True)
    D.flush()
    E.flush()
    json.dump(dict(mesh=mesh, pitch=a, K=K, N=N, rawL_end=int(rawL[-1]), rawP_end=int(rawP[-1]), rawR_end=int(rawR[-1]),
                   rows=rows), open(out / f"vals_{mesh}.json", "w"))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
