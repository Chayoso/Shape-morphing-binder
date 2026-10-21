"""r4_extra.py OUT_JSONL PNG_DIR MESH:LABEL=ARCHIVE[:ind] [...] -- D145 reading 4 (2), observed, not gated: where an end
state's display topology departs from the target's (teapot genus 0 against 1, the dragon's 2 against a perfect
sample's 9-12, the bunny target's two internal voids the ends do not have). CPU; scipy, matplotlib.

Per state x (the archive's last finite frame; with the suffix `:ind` D90's independent sample instead, the floor)
against the archive's own target sample t (pitch a = the harness's, from the independent sample):
  extra   = the points of x farther than tau * a from every point of t (material where the target has none: a web or a
            filled hole or void), missing = the points of t farther than tau * a from every point of x (a cut or an
            opened gap), tau = 1 and 2;
  their clusters at 1.5-pitch links (tau = 1): size, centroid in the target's box (0..1 per axis), extent (pitches),
  the mean distance to the other set (pitches), and, for the extra clusters, the kind of the target display surface's
  component nearest the centroid (r4/display_floor/MESH_L97_tgt_mesh.npz: `outer` = the main surface, `void` = an
  internal cavity of the target);
  a picture (three orthographic projections: t grey, extra red, missing blue) in PNG_DIR."""
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

B = "/data/relcfd/chayo/physmorph_v2"
X = f"{B}/output/gpu/d145_ext/r4"
NT = 6


def last_frame(path):
    z = np.load(path, allow_pickle=True)
    F = z["frames"]
    k = len(F) - 1
    while k > 0 and not np.isfinite(np.asarray(F[k][:16])).all():
        k -= 1
    return np.asarray(F[k], np.float64), np.asarray(z["tgt"], np.float64)


def clusters(p, a):
    if len(p) == 0:
        return np.zeros(0, int), 0
    pairs = cKDTree(p).query_pairs(1.5 * a, output_type="ndarray")
    n, lab = connected_components(coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
                                             shape=(len(p), len(p))), directed=False)
    return lab, n


def surface_kinds(mesh):
    z = np.load(f"{X}/display_floor/{mesh}_L97_tgt_mesh.npz")
    V, F = np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    nv, vlab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))),
                                    directed=False)
    main = int(np.argmax(np.bincount(vlab)))
    tri = V[F]
    sv = np.bincount(vlab[F[:, 0]], weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0,
                     minlength=nv)
    s = 1.0 if sv[main] >= 0 else -1.0
    kind = np.array(["outer" if c == main else ("void" if s * sv[c] < 0 else "piece") for c in range(nv)])
    return cKDTree(V), vlab, kind


out, png = sys.argv[1], sys.argv[2]
ref = {}
for spec in sys.argv[3:]:
    key, path = spec.split("=", 1)
    mesh, label = key.split(":")
    use_ind = path.endswith(":ind")
    path = path[:-4] if use_ind else path
    if mesh not in ref:
        I = np.asarray(np.load(f"{B}/output/gpu/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
        a = 0.708 * float(np.median(cKDTree(I).query(I[::20], 9, workers=NT)[0][:, 8]))
        ref[mesh] = (I, a, surface_kinds(mesh))
    I, a, (tv, vlab, kind) = ref[mesh]
    x, t = last_frame(path)
    if use_ind:
        x, label = I, "ind"
    dx = cKDTree(t).query(x, 1, workers=NT)[0] / a
    dt = cKDTree(x).query(t, 1, workers=NT)[0] / a
    lo, hi = t.min(0), t.max(0)
    row = dict(mesh=mesh, label=label, pitch=a, n=len(x),
               extra={f"{tau:g}": int((dx > tau).sum()) for tau in (1, 2)},
               missing={f"{tau:g}": int((dt > tau).sum()) for tau in (1, 2)})
    for name, p, d in (("extra", x[dx > 1], dx[dx > 1]), ("missing", t[dt > 1], dt[dt > 1])):
        lab, n = clusters(p, a)
        cl = []
        for c in np.argsort(-np.bincount(lab, minlength=n))[:8] if n else []:
            q, dq = p[lab == c], d[lab == c]
            e = dict(size=int(len(q)), centroid=[round(float(v), 3) for v in (q.mean(0) - lo) / (hi - lo)],
                     extent_a=[round(float(v), 1) for v in (q.max(0) - q.min(0)) / a],
                     mean_dist_a=round(float(dq.mean()), 2), max_dist_a=round(float(dq.max()), 2))
            if name == "extra":
                e["near_target_surface"] = str(kind[vlab[tv.query(q.mean(0))[1]]])
            cl.append(e)
        row[f"{name}_clusters"] = cl
        row[f"{name}_n_clusters"] = int(n)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ts = t[:: max(1, len(t) // 40000)]
        for j, (u, v) in enumerate(((0, 1), (0, 2), (2, 1))):
            ax[j].scatter(ts[:, u], ts[:, v], s=0.2, c="0.75", lw=0)
            ax[j].scatter(t[dt > 1][:, u], t[dt > 1][:, v], s=1.5, c="tab:blue", lw=0, label="missing (target, no state)")
            ax[j].scatter(x[dx > 1][:, u], x[dx > 1][:, v], s=1.5, c="tab:red", lw=0, label="extra (state, no target)")
            ax[j].set_aspect("equal")
            ax[j].set_title(f"axes {'xyz'[u]}-{'xyz'[v]}")
        ax[0].legend(loc="upper left", markerscale=6)
        fig.suptitle(f"{mesh} {label}: extra {row['extra']['1']} / missing {row['missing']['1']} at 1 pitch")
        fig.tight_layout()
        fig.savefig(f"{png}/{mesh}_{label}_extra.png", dpi=80)
        plt.close(fig)
    except Exception as ex:  # noqa: BLE001
        print("png failed", ex, flush=True)
