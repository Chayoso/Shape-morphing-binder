"""stamp.py STATE_NPZ OUT_PREFIX [+opts] -- D165, the stamp test: a plaque pressed, word face down, into a soft slab on the
floor, held, lifted; the slab keeps what the plaque's particles printed into it (the frozen material's F smoothing keeps
95.5 % of every step's deformation as flow).

STATE_NPZ: a run's end state (pipeline_run --save_end_state: x, F, Fp, vol0, ...), or a kept-frames archive (frames12: the
last frame, F = Fp = I, the end shape taken as the rest shape; +vpp the volume per particle, default the D163 plaque's
43.742 wu^3 / N). The plaque's front face (+z of the run's frame, the word SIGGRAPH) is turned to face down: stamp
(x, y, z) = (x, -z, y) of the run's frame, then shifted up only (x and z stay the run's, so the word's map is the run's).
F and Fp are turned with it (R F R^T, R Fp R^T). Each particle is split into 8 children at +-s/2 (s the children's
spacing, (vpp / 8)^(1/3)), carrying its F, Fp and an eighth of its rest volume, so that the grid of the test (dx = 2 s,
8 per cell) is twice as fine as the run's particle pitch: the carving (about one run pitch) is then about one cell. The
slab: a lattice at the same spacing s under the plaque's footprint with +margin wu around, +slab wu thick, its top at
y = 0, on the floor. Materials: the fixed-corotated model of the runs, E_plaque = 7 x 1.4e5 (D147's stiff material),
E_slab = 1.4e5 / 20, Poisson 0.2, density 800 for both, gravity +g; the frozen MPMParams otherwise (drag 0.9, smoothing
0.955); Coulomb 0.5 between the bodies (Bardenhagen, no adhesion) and on the floor (mpm/bodies.py, D147).

The plaque is held by a hand: at every window start (+T steps of +dt) its mean velocity is replaced by the hand's, its
deformation velocity kept. Stages: settle (+settle s, the hand still: the run's controls are gone, the plaque relaxes to
what its F and Fp hold), press (+v_press wu/s down until its face plane, the median over the face's columns of the
lowest particle, is +depth wu below the slab's top), hold (+hold s), lift (+v_lift wu/s up for +lift s), rest (+rest s,
the hand still above). Written: OUT_PREFIX_stamp.npz (the kept frames in scripts/render_scene.py's form: the plaque as
its parents, the 8 children's mean, and every 8th slab particle; rows, floor_y, tgt0, tgt1, raws, times, stages) and
OUT_PREFIX_end.npz (the slab's particles at the end, the plaque's parents as loaded, after the settle and at the end,
the transform and the numbers); one JSON line a window on stdout."""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import physmorph  # noqa: E402,F401  (before torch: the CuPy / NVRTC order)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from physmorph.mpm.bodies import Bodies, FloorTrajectory  # noqa: E402
from physmorph.mpm.constitutive import lame  # noqa: E402
from physmorph.mpm.state import MPMParams  # noqa: E402
from physmorph.mpm.traj import Trajectory  # noqa: E402

opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv[1:] if not s.startswith("+")]
STATE, OUT = argv[0], argv[1]
f = lambda k, d: float(opts.get(k, d))  # noqa: E731
E0 = 1.4e5
E_P, E_S, NU, RHO, G = f("E_plaque", 7 * E0), f("E_slab", E0 / 20), f("poisson", 0.2), f("rho", 800.0), f("g", 1.0)
SPLIT = int(opts.get("split", 2))                # children per axis: SPLIT^3 per particle, the grid at 2 / SPLIT of a0
CROP = f("crop", 0.0)                           # > 0: only the plaque within CROP wu of its face (a hand-held face plate)
BOX = [float(v) for v in opts["box"].split(",")] if "box" in opts else None   # x0,x1,z0,z1: only the plaque over it
DT, TW = f("dt", 2.5e-4 * 2.0 / SPLIT), int(opts.get("T", 10))
THICK, MARGIN, DEPTH, GAP0 = f("slab", 0.5), f("margin", 0.4), f("depth", 0.06), f("gap", 0.1)
V_PRESS, V_LIFT = f("v_press", 0.5), f("v_lift", 1.0)
SETTLE, HOLD, LIFT, REST = f("settle", 0.1), f("hold", 0.1), f("lift", 0.3), f("rest", 0.15)
SAVE_EVERY = int(opts.get("save_every", 4))
MU_C = f("mu", 0.5)
dev = "cuda"
t_start = time.time()

# ── the plaque as loaded ──────────────────────────────────────────────────────────────────────────────────────────────
z = np.load(STATE, allow_pickle=True)
if "F" in z.files:
    X = np.asarray(z["x"], np.float64)
    F = np.asarray(z["F"], np.float64)
    Fp = np.asarray(z["Fp"], np.float64)
    vol0 = np.asarray(z["vol0"], np.float64) if "vol0" in z.files else None
    kind = "end state (x, F, Fp)"
else:
    X = np.asarray(z["frames"][-1], np.float64)
    F = Fp = vol0 = None
    kind = "kept frame (F = Fp = I)"
ok = np.isfinite(X).all(1)
if F is not None:
    ok &= np.isfinite(F).reshape(len(F), -1).all(1) & np.isfinite(Fp).reshape(len(Fp), -1).all(1)
X = X[ok]
N = len(X)
if F is not None:
    F, Fp = F[ok], Fp[ok]
    vol0 = vol0[ok] if vol0 is not None else None
detF = np.linalg.det(F) if F is not None else np.ones(N)
if vol0 is not None:
    vpp = float(np.median(vol0 * detF))          # the volume per particle where the run ends
else:
    vpp = f("vpp", 43.742 / N)
    vol0 = np.full(N, vpp)
R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])    # (x, y, z) -> (x, -z, y): the front face down
X = X @ R.T
if F is not None:
    F = np.einsum("ij,njk,lk->nil", R, F, R)
    Fp = np.einsum("ij,njk,lk->nil", R, Fp, R)
s = vpp ** (1.0 / 3.0) / SPLIT                 # the children's spacing
dx = 2.0 * s
a0 = vpp ** (1.0 / 3.0)


def face_plane(P, h):
    """The median over the face's columns (bins of h in x, z) of the lowest particle: the plaque's face plane."""
    ij = np.floor(P[:, [0, 2]] / h).astype(np.int64)
    ij -= ij.min(0)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    order = np.argsort(key)
    ks, ys = key[order], P[order, 1]
    _, start = np.unique(ks, return_index=True)
    return float(np.median(np.minimum.reduceat(ys, start)))


if CROP > 0.0 or BOX is not None:               # the face plate: the parents within CROP of the face (and over BOX)
    keep_p = np.ones(N, bool)
    if CROP > 0.0:
        keep_p &= X[:, 1] < face_plane(X, 2.0 * a0) + CROP
    if BOX is not None:
        keep_p &= (X[:, 0] > BOX[0]) & (X[:, 0] < BOX[1]) & (X[:, 2] > BOX[2]) & (X[:, 2] < BOX[3])
    X, vol0 = X[keep_p], vol0[keep_p]
    F, Fp = (F[keep_p], Fp[keep_p]) if F is not None else (None, None)
    N = len(X)
K3 = SPLIT ** 3
g1 = (np.arange(SPLIT) + 0.5 - 0.5 * SPLIT) * s
offs = np.array([[i, j, k] for i in g1 for j in g1 for k in g1], np.float64)
XC = (X[:, None, :] + offs[None]).reshape(-1, 3)
face0 = face_plane(XC, 2.0 * a0)
shift = GAP0 - face0
X[:, 1] += shift
XC[:, 1] += shift
NC = len(XC)
FC = np.repeat(F, K3, axis=0).astype(np.float32) if F is not None else None
FpC = np.repeat(Fp, K3, axis=0).astype(np.float32) if Fp is not None else None
volC = np.repeat(vol0 / K3, K3).astype(np.float32)

# ── the slab ──────────────────────────────────────────────────────────────────────────────────────────────────────────
lo_xz = XC[:, [0, 2]].min(0) - MARGIN
hi_xz = XC[:, [0, 2]].max(0) + MARGIN
nsx, nsz = (int(math.ceil((hi_xz[k] - lo_xz[k]) / s)) for k in (0, 1))
nsy = max(2, int(round(THICK / s)))
ii, jj, kk = np.meshgrid(np.arange(nsx), np.arange(nsy), np.arange(nsz), indexing="ij")
XS = np.stack([lo_xz[0] + (ii + 0.5) * s, -(nsy - jj - 0.5) * s, lo_xz[1] + (kk + 0.5) * s], -1).reshape(-1, 3)
dec = ((ii % 2 == 0) & (jj % 2 == 0) & (kk % 2 == 0)).reshape(-1)
top = (jj == nsy - 1).reshape(-1)
NS = len(XS)
floor_y = -nsy * s
lift_room = V_LIFT * LIFT + 0.1
lo = np.array([lo_xz[0] - 4 * dx, floor_y - 4 * dx, lo_xz[1] - 4 * dx])
hi = np.array([hi_xz[0] + 4 * dx, XC[:, 1].max() + lift_room + 4 * dx, hi_xz[1] + 4 * dx])
n = [int(math.ceil((hi[k] - lo[k]) / dx)) for k in range(3)]
prm = MPMParams(dx=dx, dt=DT, f_ext=(0.0, -G, 0.0), grid_min=tuple(float(v) for v in lo), nx=n[0], ny=n[1], nz=n[2],
                floor_y=floor_y, floor_friction=MU_C)
lamP, muP = lame(E_P, NU)
lamS, muS = lame(E_S, NU)
vc = s ** 3
trP = Trajectory(XC.astype(np.float32), np.full(NC, RHO * vc, np.float32), lamP, muP, prm, TW, Fp=FpC, F0=FC,
                 vol0=volC, device=dev, requires_grad=False, persistent=True)
trS = FloorTrajectory(XS.astype(np.float32), np.full(NS, RHO * vc, np.float32), lamS, muS, prm, TW,
                      floor_c=floor_y + 0.5 * s, vol0=np.full(NS, vc, np.float32), device=dev, requires_grad=False,
                      persistent=True)
bod = Bodies([trP, trS], [vc, vc], MU_C)
graph = bod.capture()
info = dict(state=STATE, kind=kind, N=N, split=SPLIT, crop=CROP, box=BOX, children=NC, slab=NS, vpp=vpp, a0=a0, s=s, dx=dx, grid=n, dt=DT, T=TW,
            E_plaque=E_P, E_slab=E_S, poisson=NU, rho=RHO, g=G, depth=DEPTH, gap=GAP0, v_press=V_PRESS, v_lift=V_LIFT,
            stages=dict(settle=SETTLE, hold=HOLD, lift=LIFT, rest=REST), shift_y=shift, face0=face0 + shift,
            detF=[float(np.quantile(detF, q)) for q in (0.01, 0.5, 0.99)], graph=bool(graph), floor_y=floor_y)
print(json.dumps({"setup": info}), flush=True)

T_ = lambda a: wp.to_torch(a)  # noqa: E731
xP0 = T_(trP.x[0]).clone()
y_ref = float(xP0[:, 1].mean())
face_ref = face0 + shift                         # the face plane at the window start's mean height y_ref
parents_load = X.astype(np.float32).copy()
dec_t = torch.as_tensor(np.nonzero(dec)[0], device=xP0.device)


def parents():
    return T_(trP.x[0]).view(N, K3, 3).mean(1)


frames, raws, times, stages = [], [], [], []
parents_settle = None


def keep(w, t, stage):
    frames.append(torch.cat([parents(), T_(trS.x[0])[dec_t]]).cpu().numpy())
    raws.append(w * TW)
    times.append(t)
    stages.append(stage)


stage, t, w, t_stage = "settle", 0.0, 0, 0.0
end_press = None
max_windows = int((SETTLE + (GAP0 + DEPTH) / V_PRESS * 2.0 + HOLD + LIFT + REST) / (TW * DT)) + 10
keep(0, 0.0, stage)
while stage != "done" and w < max_windows:
    face = face_ref + float(T_(trP.x[0])[:, 1].mean()) - y_ref
    if stage == "settle" and t >= SETTLE:
        parents_settle = parents().cpu().numpy()
        stage, t_stage = "press", t
    if stage == "press" and face <= -DEPTH:
        stage, t_stage, end_press = "hold", t, t
    if stage == "hold" and t - t_stage >= HOLD:
        stage, t_stage = "lift", t
    if stage == "lift" and t - t_stage >= LIFT:
        stage, t_stage = "rest", t
    if stage == "rest" and t - t_stage >= REST:
        stage = "done"
        break
    hand = {"settle": 0.0, "press": -V_PRESS, "hold": 0.0, "lift": V_LIFT, "rest": 0.0}[stage]
    v0 = T_(trP.v[0])
    v0 -= v0.mean(0, keepdim=True)
    v0[:, 1] += hand
    bod.run()
    for tr in (trP, trS):
        for name in ("x", "v", "C", "F"):
            wp.copy(getattr(tr, name)[0], getattr(tr, name)[TW])
    w += 1
    t = w * TW * DT
    if w % 20 == 0 or stage != "press":
        xs, xp = T_(trS.x[0]), T_(trP.x[0])
        fin = bool(torch.isfinite(xs).all()) and bool(torch.isfinite(xp).all())
        if w % 20 == 0:
            print(json.dumps({"w": w, "t": round(t, 4), "stage": stage, "face": round(face, 5),
                              "slab_top_min": round(float(xs[torch.as_tensor(np.nonzero(top)[0], device=xs.device), 1].min()), 5),
                              "plaque_vmax": round(float(T_(trP.v[0]).norm(dim=1).max()), 4),
                              "slab_vmax": round(float(T_(trS.v[0]).norm(dim=1).max()), 4), "finite": fin}), flush=True)
        if not fin:
            print(json.dumps({"abort": "non-finite state", "w": w}), flush=True)
            break
    if w % SAVE_EVERY == 0:
        keep(w, t, stage)
keep(w, t, stage)
parents_end = parents().cpu().numpy()
slab_end = T_(trS.x[0]).cpu().numpy()
info.update(windows=w, t_end=t, end_press=end_press, seconds=time.time() - t_start, stage_end=stage)
fr = np.stack(frames).astype(np.float32)
np.savez(f"{OUT}_stamp.npz", frames=fr, rows=np.array([[0, N], [N, N + int(dec.sum())]]), floor_y=np.float64(floor_y),
         tgt0=fr[0, :N], tgt1=fr[0, N:], raws=np.asarray(raws), times=np.asarray(times), stages=np.asarray(stages))
np.savez(f"{OUT}_end.npz", slab_end=slab_end, slab_start=XS.astype(np.float32), slab_top=top,
         parents_load=parents_load, parents_settle=parents_settle if parents_settle is not None else parents_load,
         parents_end=parents_end, R=R, info=np.str_(json.dumps(info)))
print(json.dumps({"done": info}), flush=True)
