"""B12 (D145): FlexiCubes (Shen et al., TOG 42(4) 2023) from the source sphere to a target, render-only, with the
authors' code: flexicubes.py (the paper's original implementation, Apache 2.0) driven as examples/optimize.py does
(its grid, parameters, optimiser, learning-rate schedule and regularisers), with three stated changes for this
comparison (pre-registered in docs/experiments.md D145 B12):
  (1) the start is the source sphere's signed distance on the grid (|x - c| - r, negative inside as FlexiCubes
      reads it), not the example's random field;
  (2) the supervision is the pipeline's image model from the arm's fixed views (B12 Renderer): the example's mask
      term on the antialiased silhouette, and its depth term's form ((((a - b) * mask_t)^2).sum(-1) + 1e-8).sqrt()
      .mean() x 10 applied to the headlight shading in place of depth (no depth, no normals);
  (3) the example's 3D SDF loss (ground-truth signed distances at random points, kaolin) is OFF: it is 3D
      supervision, not rendering. The developability regulariser stays off (the example's default).
Example defaults kept: 64^3 voxel grid scaled x 2 over the normalised target (centred on its box, largest extent
1.8, as util.load_mesh), Adam lr 0.01 on (sdf, weight, deform), the LambdaLR 10^(-0.0002 it), the SDF regulariser
0.2 falling to 0.01 over the first quarter, L_dev x 0.5, |weight| x 0.1; 1250 iterations (the README's longer
example); every view of the arm in each iteration (the example draws 8 random views a batch).

  b12_fc.py MESH ARM SEED [--smoke K]
Writes results/fc/<mesh>_<arm>_s<seed>/: run.json, frames.npz (21 frame meshes extracted with training=False, in the
pipeline frame), curve.npz (losses; topology every 10 iterations)."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b12_common import B12, Renderer, extent_of, face_normals, pick_frames, save_json, signed_volume, source_sphere, target_mesh, topo_np, view_set  # noqa: E402

sys.path.insert(0, f"{B12}/src/FlexiCubes-HEAD")
from flexicubes import FlexiCubes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("arm", choices=["a", "b"])
ap.add_argument("seed", type=int)
ap.add_argument("--smoke", type=int, default=0)
args = ap.parse_args()

RES_GRID, LR, ITERS, SDF_REG = 64, 0.01, 1250, 0.2
if args.smoke:
    ITERS = args.smoke
tag = f"{args.mesh}_{args.arm}_s{args.seed}"
out = f"{B12}/results/{'smoke_fc' if args.smoke else 'fc'}/{tag}"
os.makedirs(out, exist_ok=True)
dev = "cuda"
torch.manual_seed(args.seed)
np.random.seed(args.seed)
t_all = time.time()


def lr_schedule(it):
    return max(0.0, 10 ** (-it * 0.0002))


def sdf_reg_loss(sdf, all_edges):
    """examples/loss.py sdf_reg_loss, verbatim (that module imports torch_scatter for the unused developability
    term, so the function is copied here)."""
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    return torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
        torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())


tm, trep = target_mesh(args.mesh)
c_src, r_src = source_sphere(args.mesh)
E = extent_of(args.mesh)
vs = view_set(args.arm, args.seed)
R = Renderer(vs, E, boost=1.0)               # nvdiffrast's default antialias gradient, as render_mesh_paper

# the example's normalisation (util.load_mesh): centre on the target's box, largest extent 1.8
vmin, vmax = tm.vertices.min(0), tm.vertices.max(0)
center = (vmax + vmin) / 2
scale = 1.8 / float((vmax - vmin).max())
center_t = torch.tensor(center, dtype=torch.float32, device=dev)
vt = torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=dev)
ft = torch.tensor(np.asarray(tm.faces), dtype=torch.int32, device=dev)
with torch.no_grad():
    ref, ref_shade, ref_cov = R.render(vt, ft, face_normals(vt, ft), per_face=True, raw=True)

fc = FlexiCubes(dev)
x_nx3, cube_fx8 = fc.construct_voxel_grid(RES_GRID)
x_nx3 *= 2
c_n = torch.tensor((c_src - center) * scale, dtype=torch.float32, device=dev)
r_n = float(r_src * scale)
sphere_reach = float(np.abs((c_src - center) * scale).max() + r_n)
sdf = torch.nn.Parameter(((x_nx3 - c_n).norm(dim=1) - r_n).clone().detach(), requires_grad=True)
weight = torch.nn.Parameter(torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device=dev), requires_grad=True)
deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)
all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
grid_edges = torch.unique(all_edges, dim=0)
optimizer = torch.optim.Adam([sdf, weight, deform], lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x))


def extract(training):
    grid_verts = x_nx3 + (2 - 1e-8) / (RES_GRID * 2) * torch.tanh(deform)
    v, f, L_dev = fc(grid_verts, sdf, cube_fx8, RES_GRID, beta_fx12=weight[:, :12], alpha_fx8=weight[:, 12:20],
                     gamma_f=weight[:, 20], training=training)
    return v / scale + center_t, f, L_dev          # back to the pipeline frame


def image_terms(v, f):
    img, shade, _ = R.render(v, f.int(), face_normals(v, f), per_face=True, raw=True)
    mask_loss = (img[..., 1:2] - ref[..., 1:2]).abs().mean()
    shade_loss = ((((shade - ref_shade) * ref_cov) ** 2).sum(-1) + 1e-8).sqrt().mean() * 10
    return mask_loss, shade_loss


Ls, cand, store, topo_curve, loss_curve = [], [], [], [], []


def record(it):
    with torch.no_grad():
        v, f, _ = extract(False)
        if f.shape[0] == 0:
            L = float("nan")
        else:
            m, s = image_terms(v, f)
            L = float(m + s)
    Ls.append(L)
    cand.append(it)
    vn, fn = v.cpu().numpy().astype(np.float32), f.cpu().numpy().astype(np.int32)
    store.append((vn, fn))
    if it % 10 == 0:
        topo_curve.append(dict(it=it, **topo_np(vn, fn)))


torch.cuda.synchronize()
t0 = time.time()
record(0)
for it in range(ITERS):
    optimizer.zero_grad()
    v, f, L_dev = extract(True)
    mask_loss, shade_loss = image_terms(v, f)
    t_iter = it / ITERS
    sdf_weight = SDF_REG - (SDF_REG - SDF_REG / 20) * min(1.0, 4.0 * t_iter)
    reg_loss = sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight
    reg_loss = reg_loss + L_dev.mean() * 0.5
    reg_loss = reg_loss + (weight[:, :20]).abs().mean() * 0.1
    total = mask_loss + shade_loss + reg_loss
    total.backward()
    optimizer.step()
    scheduler.step()
    loss_curve.append((it, float(mask_loss), float(shade_loss), float(reg_loss)))
    record(it + 1)
torch.cuda.synchronize()
wall_opt = time.time() - t0

Lf = np.where(np.isfinite(Ls), Ls, np.nanmax(Ls))
iters, q, picks = pick_frames(Lf, cand)
fr = {}
for j, i in enumerate(picks):
    fr[f"v{j:02d}"], fr[f"f{j:02d}"] = store[i]
topo_frames = [dict(topo_np(fr[f"v{j:02d}"], fr[f"f{j:02d}"]), signed_volume=signed_volume(fr[f"v{j:02d}"], fr[f"f{j:02d}"]))
               for j in range(len(picks))]
last = len(Ls) - 1
conv = (Lf[last - 100] - Lf[last]) / max(Lf[last - 100], 1e-30) if last >= 100 else None
if True:  # smoke runs save too (results/smoke*), so the CPU readout can be tested
    np.savez_compressed(f"{out}/frames.npz", raws=np.array(iters), q=np.array(q), **fr)
    np.savez_compressed(f"{out}/curve.npz", cand=np.array(cand), L=np.array(Ls), terms=np.array(loss_curve))
res = dict(method="flexicubes", mesh=args.mesh, arm=args.arm, seed=args.seed, views=vs["desc"], n_views=vs["n"],
           res=vs["res"], extent=E, target=trep, c_src=c_src.tolist(), r_src=r_src,
           normalisation=dict(center=center.tolist(), scale=scale, sphere_reach_in_grid=sphere_reach,
                              grid_half_width=1.0),
           params=dict(grid=RES_GRID, lr=LR, iterations=ITERS, sdf_regularizer=SDF_REG, L_dev=0.5, weight_l1=0.1,
                       sdf_loss="off (3D supervision)", depth="none", normals="none",
                       image_terms="mask L1 (antialiased) + 10 x masked shading (the depth term's form)",
                       views_per_iteration="all of the arm"),
           loss_start=Ls[0], loss_end=Ls[-1], loss_min=float(np.nanmin(Ls)), converged_drop_last100=conv,
           frames=dict(iterates=iters, q=q, topology=topo_frames), topology_every10=topo_curve,
           wall=dict(optimisation=wall_opt, total=time.time() - t_all),
           gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, smoke=bool(args.smoke))
save_json(f"{out}/run.json", res)
print(f"[{tag}] {ITERS} iterations, image loss {Ls[0]:.4e} -> {Ls[-1]:.4e}, last-100 drop {conv}, sphere reach "
      f"{sphere_reach:.3f} of 1, wall {wall_opt:.1f} s, peak {res['gpu_peak_gb']:.2f} GB; end topology "
      f"{topo_frames[-1]}", flush=True)
