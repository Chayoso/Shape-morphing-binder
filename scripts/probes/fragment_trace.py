"""Fragment trace: for the particles the fragment mask flags at the END of an archive, the
mean distance to their K=8 SOURCE neighbours over the frames (does the re-coupling projection
ever bring them back, or are they re-ejected every window?). Also the first frame at which
each of them became a fragment. Usage: fragment_trace.py <archive.npz> [dx]"""
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage


def fragment_mask(x, gmin, dx, dims):
    ijk = np.floor((x - gmin) / dx).astype(np.int64)
    ok = ((ijk >= 0) & (ijk < dims)).all(1)
    occ = np.zeros(dims, bool)
    occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] = True
    occ_d = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 3), bool))
    lab, n = ndimage.label(occ_d, structure=np.ones((3, 3, 3), int))
    if n <= 1:
        return np.zeros(len(x), bool)
    sizes = np.bincount(lab.ravel())[1:]
    body = 1 + int(np.argmax(sizes))
    frag = np.ones(len(x), bool)
    frag[ok] = lab[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] != body
    return frag


def main():
    d = np.load(sys.argv[1], allow_pickle=True)
    frames = d["frames"] if "frames" in d else d["x"]
    frames = np.asarray(frames, np.float32)
    n_f, N, _ = frames.shape
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else float(d["dx"]) if "dx" in d else 0.215
    x0 = frames[0]
    lo = frames.min((0, 1)) - 2 * dx
    hi = frames.max((0, 1)) + 2 * dx
    dims = np.ceil((hi - lo) / dx).astype(int) + 1
    nbr = cKDTree(x0).query(x0, k=9, workers=-1)[1][:, 1:]
    rest = np.linalg.norm(x0[nbr] - x0[:, None, :], axis=2)
    masks = [fragment_mask(frames[i], lo, dx, dims) for i in range(n_f)]
    counts = np.array([m.sum() for m in masks])
    print(f"frames {n_f} N {N} dx {dx:.3f} grid {tuple(dims)}")
    print("fragment count per 10 frames:", counts[::10].tolist())
    end = masks[-1]
    idx = np.where(end)[0]
    print(f"end fragments: {len(idx)}")
    if len(idx) == 0:
        return
    first = np.array([next((i for i in range(n_f) if masks[i][p]), -1) for p in idx])
    print("first flagged frame: min/median/max", first.min(), int(np.median(first)), first.max())
    # neighbour distance ratio (mean over the 8 source neighbours) over time, for the end fragments
    ratio = np.stack([np.linalg.norm(frames[i][nbr[idx]] - frames[i][idx][:, None, :], axis=2).mean(1) / rest[idx].mean(1)
                      for i in range(n_f)])                                     # (n_f, n_frag)
    print("mean nbr-distance / rest over the end fragments, per 10 frames:")
    print(" ".join(f"{v:.2f}" for v in ratio.mean(1)[::10]))
    # per-particle: does the ratio ever DECREASE by >20% between consecutive windows (a return)?
    dec = (ratio[1:] < 0.8 * ratio[:-1]).any(0)
    print(f"end fragments that ever came back by >20% between frames: {int(dec.sum())} of {len(idx)}")
    # how many of the end fragments are continuously flagged since first?
    cont = np.array([all(masks[i][p] for i in range(f, n_f)) for p, f in zip(idx, first)])
    print(f"continuously flagged since first: {int(cont.sum())} of {len(idx)}")
    # distance to the nearest NON-fragment particle at the end
    body = np.where(~end)[0]
    dnear = cKDTree(frames[-1][body]).query(frames[-1][idx], k=1)[0]
    print("end distance to nearest body particle: p50/p90/max", np.percentile(dnear, 50).round(3), np.percentile(dnear, 90).round(3), dnear.max().round(3))


if __name__ == "__main__":
    main()
