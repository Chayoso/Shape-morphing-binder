"""d148_cleanup_pull.py -- D148: is the spray cleanup's pull on the beads toward the far wall of a target gap? (CPU)

  beads LABEL FRAMES12 DUMP FMT K:X0,Y0,X1,Y1[:full] [...]   the particles of a bead seen on the side render at frame K
  set   LABEL FRAMES12 DUMP FMT X0,Y0,X1,Y1 T0 T1              the particles apart from the body near a gap node that
                                                              project into the box at T0-T1 s (D135's crevice foam)
  gate  X0,Y0,X1,Y1 W0 W1 LABEL=FRAMES12 [...]                per window, the particles the isolation gate counts
                                                              near the gap nodes projecting into the box

FMT v2: D146's per-particle hook (w_AAAA.npz: x0 at the window's start, d40 its displacement to the end, g_k * s_k each
term's gradient at the committed end); fb: D144's (d137_AAAA.npz: x at the end, x_start, g_k * s_k). Per particle and
window: the isolation gate at the window's start (objective.isolation_gate: the 8th-neighbour distance over its median,
ramp 1.2-1.8; the term acts where it is > 0), the cleanup's pull -g_spray, the target point nearest to the particle's
end position (the cleanup pulls down the target's distance field, toward the nearest occupied cell), and the gap plane
of the D148 gap node nearest the particle (gap_split.GapPlanes on the run's float32 loss cell): whether the nearest
target point lies across that plane from the particle's origin (its position at the last window before the gate first
counted it, else the first window read), and the cosine of the pull with the plane's normal turned from the origin's
side to the other (> 0: toward the far wall)."""
import json
import math
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_d148x")
from physmorph.mpm.gap_split import GapPlanes  # noqa: E402

DX, NG, LRES = 0.2948594285057975, 44, 87
GMIN = np.full(3, -DX * NG / 2.0)
D0 = GMIN.astype(np.float32)
CELL = float((D0 + DX * np.full(3, NG, np.float32) - D0).max() / LRES)
OFFS = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)


def camera(tgt):
    center = tgt.mean(0)
    radius = float(np.linalg.norm(tgt - center, axis=1).max())
    W, H, AZ, EL = 3840, 2160, math.radians(90.0), math.radians(18.0)
    cam = center + 3.6 * radius * np.array([math.cos(EL) * math.sin(AZ), math.sin(EL), math.cos(EL) * math.cos(AZ)])
    fwd = (center - cam) / np.linalg.norm(center - cam)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tan_y = math.tan(math.radians(15.0))
    tan_x = tan_y * W / H

    def pixels(p):
        d = p - cam
        xc, yc, zc = d @ right, d @ up, d @ fwd
        return np.stack((((xc / zc) / tan_x + 1) / 2 * W - 0.5, (1 - (yc / zc) / tan_y) / 2 * H - 0.5), 1)
    return pixels


def in_box(pixels, x, box, shrink=False):
    x0b, y0b, x1b, y1b = box
    wq, hq = ((x1b - x0b) / 4.0, (y1b - y0b) / 4.0) if shrink else (0, 0)
    q = pixels(x)
    return (q[:, 0] >= x0b + wq) & (q[:, 0] <= x1b - wq) & (q[:, 1] >= y0b + hq) & (q[:, 1] <= y1b - hq)


def gate(x):
    d, _ = cKDTree(x).query(x, 9, workers=16)
    r = d[:, 8] / max(float(np.median(d[:, 8])), 1e-12)
    return np.clip((r - 1.2) / 0.6, 0.0, 1.0)


def apart(x, a):
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, cl = connected_components(g, directed=False)
    return cl != np.bincount(cl).argmax()


class Run:
    def __init__(self, frames, dump, fmt):
        z = np.load(frames)
        self.fr, self.raws = z["frames"], np.asarray(z["raws"])
        self.tgt = np.asarray(z["tgt"], np.float64)
        self.ttree = cKDTree(self.tgt)
        self.a = 0.708 * float(np.median(self.ttree.query(self.tgt[::10], 9)[0][:, 8]))
        gp = GapPlanes(self.tgt, DX, GMIN, (NG,) * 3, CELL)
        self.pm, self.pn, self.mask = gp.m.reshape(-1, 3), gp.n.reshape(-1, 3), gp.mask
        self.xg = GMIN + DX * np.stack(np.unravel_index(np.arange(NG ** 3), (NG,) * 3), 1)
        self.pixels = camera(self.tgt)
        self.dump, self.fmt = dump, fmt

    def window(self, w):
        """(x at the window's start, x at its end, the cleanup's pull) of 1-based window w, or None."""
        try:
            if self.fmt == "v2":
                d = np.load(f"{self.dump}/w_{w:04d}.npz")
                if "g_spray" not in d.files:
                    return None
                x0 = d["x0"].astype(np.float64)
                xe = x0 + d["d40"].astype(np.float64)
            else:
                d = np.load(f"{self.dump}/d137_{w:04d}.npz")
                x0, xe = d["x_start"].astype(np.float64), d["x"].astype(np.float64)
            return x0, xe, -(d["g_spray"].astype(np.float64) * float(d["s_spray"]))
        except OSError:
            return None

    def node(self, x):
        """the D148 gap node in x's stencil nearest to it (-1 if none)."""
        b = np.floor((x - GMIN) / DX).astype(int) - 1
        best, bd = -1, 1e9
        for o in OFFS:
            n = b + o
            if (n < 0).any() or (n >= NG).any():
                continue
            g = int(np.ravel_multi_index(tuple(n), (NG,) * 3))
            if self.mask[g]:
                dd = float(np.linalg.norm(self.xg[g] - x))
                if dd < bd:
                    best, bd = g, dd
        return best


def read_set(run, label, pset, w0, w1):
    states = {w: run.window(w) for w in range(w0, w1 + 1)}
    states = {w: s for w, s in states.items() if s is not None}
    if not states:
        print(f"  {label}: no per-particle window in {w0}-{w1}")
        return
    gates = {w: gate(s[0]) for w, s in states.items()}
    ws = sorted(states)
    first = {}
    origin = {}
    for p in pset:
        prev = None
        for w in ws:
            if gates[w][p] > 0 and p not in first:
                first[p] = w
                origin[p] = states[prev][0][p] if prev is not None else states[w][0][p]
            prev = w
    print(f"== {label}: {len(pset)} particles; counted by the gate at some window {len(first)}; first counted at windows "
          f"{sorted(set(first.values()))[:12]}", flush=True)
    for w in ws:
        x0, xe, pull = states[w]
        rows = []
        for p in pset:
            if p not in first or w < first[p] or gates[w][p] <= 0:
                continue
            g = run.node(xe[p])
            if g < 0:
                rows.append((0, np.nan, np.nan, np.linalg.norm(pull[p]), np.nan))
                continue
            m, n = run.pm[g], run.pn[g]
            so = np.sign((origin[p] - m) @ n) or 1.0
            _, it = run.ttree.query(xe[p])
            t = run.tgt[it]
            st = np.sign((t - m) @ n)
            sx = np.sign((xe[p] - m) @ n)
            pl = pull[p]
            c = float((pl @ (-so * n)) / max(np.linalg.norm(pl), 1e-30))
            rows.append((1, float(st != so), float(st != sx), np.linalg.norm(pl), c))
        if not rows:
            continue
        r = np.array(rows, float)
        near = r[:, 0] > 0
        print(f"  window {w} (from {(w - 1) / 6:.2f} s): gated {len(r)}, near a gap node {int(near.sum())}; nearest "
              f"target point across the plane from the origin {np.nanmean(r[near, 1]) if near.any() else np.nan:.2f}, "
              f"from the particle itself {np.nanmean(r[near, 2]) if near.any() else np.nan:.2f}; the pull's cosine "
              f"with the normal toward the far side median {np.nanmedian(r[near, 4]) if near.any() else np.nan:+.2f} "
              f"(> 0.5 at {np.nanmean(r[near, 4] > 0.5) if near.any() else np.nan:.2f}, < -0.5 at "
              f"{np.nanmean(r[near, 4] < -0.5) if near.any() else np.nan:.2f}); |pull| median {np.median(r[:, 3]):.2e}",
              flush=True)


def bead_set(run, spec):
    p = spec.split(":")
    K, box, full = int(p[0]), [int(v) for v in p[1].split(",")], len(p) > 2 and p[2] == "full"
    x = np.asarray(run.fr[K], np.float64)
    return np.nonzero(in_box(run.pixels, x, box, shrink=not full))[0], K


def foam_set(run, box, t0, t1):
    pick = set()
    b0 = None
    for k in range(len(run.raws)):
        if not t0 <= run.raws[k] / 240 <= t1:
            continue
        x = np.asarray(run.fr[k], np.float64)
        ap = np.nonzero(apart(x, run.a))[0]
        ap = ap[in_box(run.pixels, x[ap], box)]
        b = np.floor((x[ap] - GMIN) / DX).astype(int) - 1
        near = np.zeros(len(ap), bool)
        m3 = run.mask.reshape((NG,) * 3)
        for o in OFFS:
            n = b + o
            ok = ((n >= 0) & (n < NG)).all(1)
            near[ok] |= m3[n[ok, 0], n[ok, 1], n[ok, 2]]
        pick |= set(ap[near].tolist())
        b0 = k
    return np.asarray(sorted(pick), np.int64)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "beads":
        label, frames, dump, fmt = sys.argv[2:6]
        run = Run(frames, dump, fmt)
        for spec in sys.argv[6:]:
            pset, K = bead_set(run, spec)
            wk = int(run.raws[K] // 40) + 1
            read_set(run, f"{label} bead {spec} (frame {K}, {run.raws[K] / 240:.2f} s; particles {pset.tolist()[:8]})",
                     pset, max(1, wk - 10), wk + 1)
    elif mode == "set":
        label, frames, dump, fmt, box, t0, t1 = sys.argv[2:9]
        run = Run(frames, dump, fmt)
        box, t0, t1 = [int(v) for v in box.split(",")], float(t0), float(t1)
        pset = foam_set(run, box, t0, t1)
        read_set(run, f"{label} set in {box} at {t0}-{t1} s", pset, max(1, int(t0 * 6) - 8), int(t1 * 6) + 1)
    else:
        box, w0, w1 = [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4])
        for spec in sys.argv[5:]:
            label, frames = spec.split("=", 1)
            run = Run(frames, None, None)
            line = []
            for w in range(w0, w1 + 1):
                k = int(np.argmin(np.abs(run.raws - 40 * (w - 1))))
                x = np.asarray(run.fr[k], np.float64)
                gt = gate(x)
                sel = np.nonzero(gt > 0)[0]
                sel = sel[in_box(run.pixels, x[sel], box)]
                near = np.array([run.node(x[p]) >= 0 for p in sel], bool) if len(sel) else np.zeros(0, bool)
                line.append(f"{w}:{int(near.sum())}")
            print(f"{label}: gated near a gap node in the box, per window (1-based: count): " + " ".join(line),
                  flush=True)
