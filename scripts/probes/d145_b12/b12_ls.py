"""B12 (D145): Large Steps in Inverse Rendering of Geometry (Nicolet, Jacobson & Jakob, TOG 40(6) 2021) from the
source sphere to a target, render-only, with the authors' code: the `largesteps` package (0.2.2, PyPI, BSD-3:
compute_matrix, to/from_differential with cholespy's Cholesky solver, AdamUniform) and the repository's geometry
helpers (scripts/geometry.py: face / vertex normals); remeshing by Botsch & Kobbelt through gpytoolbox's
remesh_botsch (the remesher of the Mitsuba 3 tutorial; the same code as the repository's submodule
sgsellan/botsch-kobbelt-remesher-libigl).

Schedule (the authors' Mitsuba 3 "Shape optimization" tutorial, translated to the PyTorch package; pre-registered in
docs/experiments.md D145 B12): a 10k icosphere (subdivision 5, 10242 vertices) at the source sphere; lambda 25,
AdamUniform lr 0.1, L1 image loss; phase 1: 100 iterations, each one step per view in view order; remesh to half the
mean edge length (5 iterations, projected); a new parameterisation and optimiser (lr 0.1); phase 2: 100 iterations of
one step on all views. Extension (pre-registered): while the phase-2 loss fell by more than 2 % over the last 20
iterations, 100 more, up to 500 phase-2 iterations. nvdiffrast's gradient boost 3 (the authors' PyTorch scripts).

  b12_ls.py MESH ARM SEED [--smoke K]     (ARM a: matched views, b: strong views; --smoke: K phase-1 iterations,
                                             10 phase-2 iterations, nothing kept)
Writes results/ls/<mesh>_<arm>_s<seed>/: run.json (config, losses, timing, convergence), frames.npz (the 21 frame
meshes in the pipeline frame, their iterate index and loss progress)."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b12_common import B12, Renderer, extent_of, pick_frames, save_json, signed_volume, source_sphere, target_mesh, topo_np, view_set  # noqa: E402

sys.path.insert(0, f"{B12}/src/large-steps-pytorch-HEAD")
from largesteps.geometry import compute_matrix  # noqa: E402
from largesteps.optimize import AdamUniform  # noqa: E402
from largesteps.parameterize import from_differential, to_differential  # noqa: E402
from scripts.geometry import compute_face_normals, compute_vertex_normals  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("arm", choices=["a", "b"])
ap.add_argument("seed", type=int)
ap.add_argument("--smoke", type=int, default=0)
args = ap.parse_args()

LAMBDA, LR, BOOST = 25.0, 0.1, 3.0
P1, P2, P2_MAX, EXT_TOL, EXT_WIN = 100, 100, 500, 0.02, 20
if args.smoke:
    P1, P2, P2_MAX = args.smoke, 10, 10
tag = f"{args.mesh}_{args.arm}_s{args.seed}"
out = f"{B12}/results/{'smoke' if args.smoke else 'ls'}/{tag}"
os.makedirs(out, exist_ok=True)
dev = "cuda"
torch.manual_seed(args.seed)
t_all = time.time()

tm, trep = target_mesh(args.mesh)
c_src, r_src = source_sphere(args.mesh)
E = extent_of(args.mesh)
vs = view_set(args.arm, args.seed)
R = Renderer(vs, E, boost=BOOST)
V = vs["n"]

vt = torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=dev)
ft = torch.tensor(np.asarray(tm.faces), dtype=torch.int32, device=dev)
with torch.no_grad():
    ref = R.render(vt, ft, compute_vertex_normals(vt, ft, compute_face_normals(vt, ft)))

import trimesh  # noqa: E402
ico = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
v0 = torch.tensor(np.asarray(ico.vertices) * r_src + c_src, dtype=torch.float32, device=dev)
f = torch.tensor(np.asarray(ico.faces), dtype=torch.int32, device=dev)


def normals(v, f):
    return compute_vertex_normals(v, f, compute_face_normals(v, f))


def full_loss(v, f):
    with torch.no_grad():
        return float((R.render(v, f, normals(v, f)) - ref).abs().mean())


cand, Ls, phase, mesh_of = [], [], [], []          # candidate iterates: step index, full L1, phase, (v, f index)
faces_by_phase = [np.asarray(ico.faces, np.int32)]
verts_store = []
step_losses = []


def record(k, v, f_idx, ph):
    cand.append(k)
    Ls.append(full_loss(v, f))
    phase.append(ph)
    verts_store.append(v.detach().cpu().numpy().astype(np.float32))
    mesh_of.append(f_idx)


torch.cuda.synchronize()
t0 = time.time()
M = compute_matrix(v0, f, lambda_=LAMBDA)
u = to_differential(M, v0).clone().detach().requires_grad_(True)
opt = AdamUniform([u], lr=LR)
k = 0
v = from_differential(M, u, "Cholesky")
record(0, v, 0, 1)
for it in range(P1):
    for vi in range(V):
        v = from_differential(M, u, "Cholesky")
        img = R.render(v, f, normals(v, f), views=[vi])
        loss = (img - ref[vi:vi + 1]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        k += 1
        step_losses.append((k, 1, vi, float(loss)))
    with torch.no_grad():
        v = from_differential(M, u, "Cholesky")
    record(k, v, 0, 1)
torch.cuda.synchronize()
t_p1 = time.time() - t0

# remesh (the tutorial: half the mean edge length, 5 iterations, projected)
from gpytoolbox import remesh_botsch  # noqa: E402
t1 = time.time()
with torch.no_grad():
    v_np = from_differential(M, u, "Cholesky").cpu().numpy().astype(np.float64)
f_np = np.asarray(ico.faces, np.int32)
l0 = np.linalg.norm(v_np[f_np[:, 0]] - v_np[f_np[:, 1]], axis=1)
l1 = np.linalg.norm(v_np[f_np[:, 1]] - v_np[f_np[:, 2]], axis=1)
l2 = np.linalg.norm(v_np[f_np[:, 2]] - v_np[f_np[:, 0]], axis=1)
h = float(np.mean([l0, l1, l2]) / 2)
v2, f2 = remesh_botsch(v_np, f_np, i=5, h=h, project=True)
t_remesh = time.time() - t1
faces_by_phase.append(np.asarray(f2, np.int32))
f = torch.tensor(np.asarray(f2, np.int32), dtype=torch.int32, device=dev)
v0b = torch.tensor(np.asarray(v2), dtype=torch.float32, device=dev)
M = compute_matrix(v0b, f, lambda_=LAMBDA)
u = to_differential(M, v0b).clone().detach().requires_grad_(True)
opt = AdamUniform([u], lr=LR)
v = from_differential(M, u, "Cholesky")
record(k, v, 1, 2)
p2_losses = []
t2 = time.time()
it2, ext_log = 0, []
target_it = P2
while it2 < target_it:
    v = from_differential(M, u, "Cholesky")
    img = R.render(v, f, normals(v, f))
    loss = (img - ref).abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    k += 1
    it2 += 1
    step_losses.append((k, 2, -1, float(loss)))
    with torch.no_grad():
        v = from_differential(M, u, "Cholesky")
    record(k, v, 1, 2)
    p2_losses.append(Ls[-1])
    if it2 == target_it and target_it < P2_MAX:
        a, b = p2_losses[-1 - EXT_WIN], p2_losses[-1]
        drop = (a - b) / max(a, 1e-30)
        ext_log.append(dict(at=it2, drop_last20=drop))
        if drop > EXT_TOL:
            target_it = min(target_it + 100, P2_MAX)
torch.cuda.synchronize()
t_p2 = time.time() - t2
wall_opt = time.time() - t0

# convergence reading (pre-registered): the relative fall of the full loss over the last 20 phase-2 iterations
conv = (p2_losses[-1 - EXT_WIN] - p2_losses[-1]) / max(p2_losses[-1 - EXT_WIN], 1e-30) if len(p2_losses) > EXT_WIN \
    else None
iters, q, picks = pick_frames(Ls, cand)
fr = {}
for j, (i, kk) in enumerate(zip(picks, iters)):
    fr[f"v{j:02d}"] = verts_store[i]
    fr[f"f{j:02d}"] = faces_by_phase[mesh_of[i]]
topo_frames = [dict(topo_np(fr[f"v{j:02d}"], fr[f"f{j:02d}"]), signed_volume=signed_volume(fr[f"v{j:02d}"], fr[f"f{j:02d}"]))
               for j in range(len(picks))]
if not args.smoke:
    np.savez_compressed(f"{out}/frames.npz", raws=np.array(iters), q=np.array(q), **fr)
    np.savez_compressed(f"{out}/curve.npz", cand=np.array(cand), L=np.array(Ls), phase=np.array(phase),
                        steps=np.array(step_losses))
res = dict(method="large_steps", mesh=args.mesh, arm=args.arm, seed=args.seed, views=vs["desc"], n_views=V,
           res=vs["res"], extent=E, target=trep, c_src=c_src.tolist(), r_src=r_src,
           start=dict(icosphere_subdivisions=5, V=int(v0.shape[0]), F=int(len(ico.faces))),
           params=dict(lambda_=LAMBDA, lr=LR, optimizer="AdamUniform", loss="L1", boost=BOOST, phase1_iterations=P1,
                       phase1="one step per view, view order", phase2_iterations=it2, phase2="one step on all views",
                       remesh=dict(h=h, iterations=5, project=True, V=int(len(v2)), F=int(len(f2)),
                                   seconds=t_remesh),
                       extension=dict(rule=f"+100 while the phase-2 loss fell > {EXT_TOL:.0%} over the last "
                                           f"{EXT_WIN} iterations, up to {P2_MAX}", log=ext_log)),
           steps_total=k, loss_start=Ls[0], loss_end=Ls[-1], loss_min=float(np.min(Ls)),
           loss_after_phase1=Ls[P1] if len(Ls) > P1 else None, converged_drop_last20=conv,
           frames=dict(iterates=iters, q=q, topology=topo_frames),
           wall=dict(phase1=t_p1, remesh=t_remesh, phase2=t_p2, optimisation=wall_opt, total=time.time() - t_all),
           gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, smoke=bool(args.smoke))
save_json(f"{out}/run.json", res)
print(f"[{tag}] steps {k} (phase 2 {it2}), loss {Ls[0]:.4e} -> {Ls[-1]:.4e} (after phase 1 {res['loss_after_phase1']}), "
      f"last-20 drop {conv}, remesh V {len(v2)} F {len(f2)}, wall {wall_opt:.1f} s, peak "
      f"{res['gpu_peak_gb']:.2f} GB; end topology {topo_frames[-1]}", flush=True)
