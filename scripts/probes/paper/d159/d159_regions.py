"""d159_regions.py OUT_DIR MESH [MESH ...] -- D159 step 0 (b): the regions of fine relief, fixed from the exact mesh before
any run (CPU; measurement only). The mesh in the run's seed-97 frame (output/gpu/d154/meshes/MESH_97.npz, mesh_frames.py:
the frame that reproduces the run's target sample), closed and turned outward (render/mesh_shell.closed_outward, D157's
preparation). The pitch a = 0.708 x the median 8th-neighbour distance of the run's target sample (B1's L97 end1), the
MPM cell from B1's JSON.
  samples   area samples of the mesh, one per (0.5 a)^2, with their faces' normals;
  graph     each sample's 16 nearest, same-side (normals within 60 degrees), Gaussian weights of one sample spacing;
            repeated weighted means (the passes for a Gaussian of sigma from the measured variance of one pass);
  relief    r = n . (p - G_2(p)): the height of the surface above its own Gaussian mean at sigma = 2 pitches (the part
            below about 10.8 pitches, D107's family), and its bands below 2.7 / 2.7-5.4 / 5.4-10.8 pitches;
  amplitude A = the local standard deviation of r over a Gaussian of 1.5 pitches (a window of about 3 pitches; the
            local mean, which carries the smoothing's curvature bias, removed);
  thin      the shape diameter along -n below two MPM cells (thin.py's threshold), by Open3D's ray caster;
  region    not thin and A in the top quarter of the not-thin samples; the rest of the not-thin samples the contrast.
Writes OUT_DIR/regions_MESH.npz (P, N, r, bands, A, T, region, contrast, thin), regions.jsonl, and maps of A and of the
region from the deliverable's front and side (regions_MESH.png)."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from physmorph.render.mesh_shell import closed_outward

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
SIGMAS = (0.5, 1.0, 2.0)
AMP = float(dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+")).get("band", "10.8"))   # the amplitude's band edge, pitches


def pitch_of(x):
    sub = x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    d = cKDTree(sub).query(sub, 9)[0][:, 8]
    return .708 * float(np.median(d)) * (20000 / len(x)) ** (1 / 3)


class Graph:
    def __init__(self, P, N, k=16):
        d, nb = cKDTree(P).query(P, k + 1)
        side = ((N[nb] * N[:, None]).sum(-1) > 0.5)
        s = float(np.median(d[:, 1]))
        w = np.exp(-0.5 * (d / s) ** 2) * side
        w[:, 0] = 1.0
        self.w = w / w.sum(1, keepdims=True)
        self.nb = nb
        self.var = float(np.median((self.w * d ** 2).sum(1))) / 2.0
        self.spacing = s

    def passes(self, sigma):
        return max(1, int(round(sigma ** 2 / self.var)))

    def smooth(self, v, n):
        for _ in range(n):
            v = (self.w[..., None] * v[self.nb]).sum(1) if v.ndim == 2 else (self.w * v[self.nb]).sum(1)
        return v


def zmap(P, N, vals, az, el, px):
    az, el = np.radians(az), np.radians(el)
    d = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    r = np.cross([0., 1., 0.], d); r /= np.linalg.norm(r)
    u = np.cross(d, r)
    X, Y, Z, F = P @ r, P @ u, P @ d, N @ d
    lo, hi = np.array([X.min(), Y.min()]), np.array([X.max(), Y.max()])
    W, H = int(np.ceil((hi[0] - lo[0]) / px)) + 1, int(np.ceil((hi[1] - lo[1]) / px)) + 1
    ok = F > 0.05
    ix, iy = ((X[ok] - lo[0]) / px).astype(int), ((hi[1] - Y[ok]) / px).astype(int)
    key = iy * W + ix
    o = np.lexsort((Z[ok], key))
    last = np.r_[key[o][1:] != key[o][:-1], True]
    sel = o[last]
    img = np.full(H * W, np.nan)
    img[key[sel]] = vals[ok][sel]
    return img.reshape(H, W)


def main():
    args = [s for s in sys.argv[1:] if not s.startswith("+")]
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    for mesh in args[1:]:
        m = np.load(G / "d154/meshes" / f"{mesh}_97.npz")
        V, F, loops, tight = closed_outward(m["V"], m["F"])
        tm = trimesh.Trimesh(V, F, process=False)
        z = np.load(G / "d145t_b1" / f"{mesh}300k_L97_end1.npz", allow_pickle=True)
        a = pitch_of(np.asarray(z["tgt"], np.float64))
        js = json.load(open(G / "d145t_b1" / f"{mesh}300k_L97.json"))
        dx = float(js["provenance"]["mpm"]["dx"])
        n = int(tm.area / (0.5 * a) ** 2)
        P, fi = trimesh.sample.sample_surface(tm, n, seed=0)
        P = np.asarray(P, np.float64)
        N = np.asarray(tm.face_normals[fi], np.float64)
        g = Graph(P, N)
        Gs, cur, done = {}, P, 0
        for s in SIGMAS:
            k = g.passes(s * a)
            cur = g.smooth(cur, k - done)
            done = k
            Gs[s] = cur
        r = (N * (P - Gs[2.0])).sum(1)
        bands = np.stack([(N * (P - Gs[0.5])).sum(1), (N * (Gs[0.5] - Gs[1.0])).sum(1), (N * (Gs[1.0] - Gs[2.0])).sum(1)], 1)
        kw = g.passes(1.5 * a)
        ra = r if AMP > 6 else bands[:, 0] + bands[:, 1]          # +band=5.4: the amplitude of the part below 5.4 pitches
        mu = g.smooth(ra, kw)
        A = np.sqrt(np.maximum(g.smooth(ra * ra, kw) - mu ** 2, 0.0))
        import open3d as o3d
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        rays = np.concatenate([P - 1e-4 * a * N, -N], 1).astype(np.float32)
        T = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
        thin = ~(T >= 2 * dx)                                      # no hit (an open spot) counts as thin
        thr = float(np.quantile(A[~thin], 0.75))
        region = (~thin) & (A >= thr)
        contrast = (~thin) & ~region
        np.savez_compressed(out / f"regions_{mesh}_b{AMP:g}.npz", P=P.astype(np.float32), N=N.astype(np.float32), r=r.astype(np.float32),
                            bands=bands.astype(np.float32), A=A.astype(np.float32), T=T.astype(np.float32), region=region,
                            contrast=contrast, thin=thin, pitch=a, dx=dx, threshold=thr)
        row = dict(mesh=mesh, amp_band=AMP, pitch=a, dx=dx, samples=n, sample_spacing_a=g.spacing / a, pass_var_a2=g.var / a ** 2,
                   passes=[g.passes(s * a) for s in SIGMAS], loops_closed=loops, watertight=tight,
                   area_a2=float(tm.area / a ** 2), thin_share=float(thin.mean()), region_share=float(region.mean()),
                   contrast_share=float(contrast.mean()), A_threshold_a=thr / a,
                   relief_rms_a={"region": float(np.sqrt((r[region] ** 2).mean()) / a),
                                 "contrast": float(np.sqrt((r[contrast] ** 2).mean()) / a),
                                 "thin": float(np.sqrt((r[thin] ** 2).mean()) / a)},
                   band_rms_region_a=[float(np.sqrt((bands[region, i] ** 2).mean()) / a) for i in range(3)],
                   band_rms_contrast_a=[float(np.sqrt((bands[contrast, i] ** 2).mean()) / a) for i in range(3)])
        with open(out / "regions.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lab = np.where(thin, 0.0, np.where(region, 2.0, 1.0))
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        for j, (az, el) in enumerate(((35, 18), (90, 10))):
            im = ax[0, j].imshow(zmap(P, N, A / a, az, el, 0.6 * a), cmap="viridis", vmin=0, vmax=float(np.quantile(A / a, 0.98)))
            ax[1, j].imshow(zmap(P, N, lab, az, el, 0.6 * a), cmap=matplotlib.colors.ListedColormap(["#c3c2b7", "#e1e0d9", "#2a78d6"]),
                            vmin=0, vmax=2)
            for i in range(2):
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[0, j].set_title(f"fine-relief amplitude below {AMP:g} pitches, view az {az} el {el}", fontsize=10)
            ax[1, j].set_title("blue: the region (top quarter of the not-thin surface); light: contrast; dark: thin", fontsize=9)
        fig.colorbar(im, ax=ax[0], fraction=0.02)
        fig.suptitle(f"{mesh}: regions fixed from the exact mesh before the D159 runs", fontsize=11)
        fig.savefig(out / f"regions_{mesh}_b{AMP:g}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
