"""Compare gradient flow: isotropic vs anisotropic across episodes."""
import numpy as np
import torch
import sys
import trimesh
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import setup_cameras, setup_renderer, compute_sigma0
from renderer.utils.covariance import _rotation_matrix_to_quaternion_torch
import torch.nn.functional as TF

device = torch.device('cuda')
cfg = yaml.safe_load(open('configs/morph_from_isosphere.yaml'))
cfg = {**cfg['defaults'], **{k: v for k, v in cfg.items() if k not in ('defaults', 'experiments')}}
rcfg = cfg.get('render', {})
cam_cfg = cfg.get('camera', {})
multi_cfg = cfg.get('multi_view', {})
color = [0.27, 0.51, 0.71]

all_cams, _, _, _ = setup_cameras(cam_cfg, multi_cfg)
cam = all_cams[0]

# Target
tgt_mesh = trimesh.load('assets/spot.obj', force='mesh')
tgt_verts = np.array(tgt_mesh.vertices, dtype=np.float32)
tgt_sigma = compute_sigma0(tgt_verts, 0.7)
N_tgt = len(tgt_verts)

r, _ = setup_renderer(cam, rcfg, training_mode=True)

with torch.no_grad():
    _, _, _, tgt_ao, _, _ = r.rasterizer(
        means3D=torch.from_numpy(tgt_verts).float().to(device),
        means2D=torch.zeros((N_tgt, 3), device=device),
        opacities=torch.full((N_tgt, 1), 0.95, device=device),
        colors_precomp=torch.tensor(color, device=device).float().unsqueeze(0).expand(N_tgt, -1),
        scales=torch.full((N_tgt, 3), tgt_sigma, device=device),
        rotations=torch.tensor([[1, 0, 0, 0]], device=device).float().expand(N_tgt, -1),
        norm3Ds_precomp=torch.zeros((N_tgt, 3), device=device).scatter_(
            1, torch.full((N_tgt, 1), 2, dtype=torch.long, device=device), 1.0))
tgt_alpha = (tgt_ao[0] if tgt_ao.dim() == 3 else tgt_ao).detach()

header = f"{'ep':>4s} | {'iso_scale':>12s} {'iso_rot':>12s} {'iso_total':>12s} | {'ani_scale':>12s} {'ani_rot':>12s} {'ani_total':>12s} | rot_ratio"
print(header)
print('-' * len(header))

episodes = [0, 1, 3, 5, 10, 15, 20, 30, 40, 59]

for ep in episodes:
    ckpt = np.load(f'output/morph_from_isosphere/isosphere_to_spot/checkpoints/ckpt_ep{ep:03d}.npz')
    x = ckpt['positions']
    F_el = ckpt['F_elastic']
    N = len(x)
    sigma0 = compute_sigma0(x, 0.7)

    results = {}
    for sname in ['iso', 'aniso']:
        if sname == 'iso':
            scales_t = torch.full((N, 3), sigma0, device=device, requires_grad=True)
            quats_t = torch.zeros((N, 4), device=device)
            quats_t[:, 0] = 1.0
            quats_t = quats_t.requires_grad_(True)
        else:
            F_t = torch.from_numpy(F_el).float().to(device)
            U, Sv, Vh = torch.linalg.svd(F_t)
            scales_t = (Sv * sigma0).requires_grad_(True)
            quats_t = _rotation_matrix_to_quaternion_torch(U @ Vh).requires_grad_(True)
            del F_t, U, Sv, Vh

        x_t = torch.from_numpy(x).float().to(device)
        m2d = torch.zeros((N, 3), device=device, requires_grad=True)
        co, _, _, ao, _, _ = r.rasterizer(
            means3D=x_t, means2D=m2d,
            opacities=torch.full((N, 1), 0.95, device=device),
            colors_precomp=torch.tensor(color, device=device).float().unsqueeze(0).expand(N, -1),
            scales=scales_t,
            rotations=TF.normalize(quats_t, p=2, dim=1),
            norm3Ds_precomp=torch.zeros((N, 3), device=device).scatter_(
                1, torch.full((N, 1), 2, dtype=torch.long, device=device), 1.0))
        alpha = ao[0] if ao.dim() == 3 else ao
        loss = ((alpha - tgt_alpha) ** 2).mean()
        loss.backward()

        gs = scales_t.grad.norm(dim=1).sum().item() if scales_t.grad is not None else 0
        gr = quats_t.grad.norm(dim=1).sum().item() if quats_t.grad is not None else 0
        results[sname] = {'scale': gs, 'rot': gr, 'total': gs + gr}
        torch.cuda.empty_cache()

    rot_ratio = results['aniso']['rot'] / max(results['iso']['rot'], 1e-15)
    print(f"{ep:4d} | {results['iso']['scale']:12.6f} {results['iso']['rot']:12.6f} {results['iso']['total']:12.6f} "
          f"| {results['aniso']['scale']:12.6f} {results['aniso']['rot']:12.6f} {results['aniso']['total']:12.6f} "
          f"| {rot_ratio:.1f}x")
