"""
Diagnostic: Measure Hessian amplification of render dL/dF through C++ backward.

1. Run physics pass 1 → note gradient norm (physics-only)
2. Compute render dL/dF at final state
3. Inject render dL/dF via set_render_gradients
4. Run physics pass 2 (skip_setup) → note gradient norm (physics+render)
5. Compare: amplification = (combined_norm - physics_norm) / render_inject_norm

Usage:
    python tools/diagnose_hessian_amp.py -c configs/sphere_to_bunny.yaml
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import torch.nn.functional as F_nn
import yaml
from pathlib import Path

from sampling.geometry.deformation_covariance import build_deformation_covariance
from sampling.analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
from utils.physics_utils import (
    build_opt_input,
    initialize_point_clouds,
    initialize_grids,
    initialize_comp_graph,
)
from utils.rendering_utils import setup_renderer, generate_target_render, prepare_rendering_inputs
from sampling import default_cfg, upsample
from sampling.utils.config_adapter import adapt_config


def compute_render_grads(x, F_e, target_alpha, renderer, campos, render_cfg, particle_color, sigma0=0.2, sv_min=0.3):
    """Compute dL/dx and dL/dF from differentiable render."""
    knn = HybridFAISSKNN(use_faiss=FAISS_AVAILABLE, tau=0.15)
    cov_cfg = {"sigma0": sigma0, "k_F": 32, "use_polar_decomposition": True,
               "use_F_smoothing": False, "sv_min": sv_min}

    x_t = torch.from_numpy(x).requires_grad_(True)
    F_t = torch.from_numpy(F_e).requires_grad_(True)

    cov, _, _ = build_deformation_covariance(
        points=x_t, x_low=x_t, F_low=F_t, knn=knn, cfg=cov_cfg)

    with torch.no_grad():
        normals_approx = torch.nn.functional.normalize(F_t.detach()[:, :, 2], dim=-1)
    from renderer import compute_shading
    rgb_np = compute_shading(x_t.detach().cpu().numpy(), normals_approx.cpu().numpy(),
                             camera_pos=campos, light_cfg=render_cfg.get("lighting", {}),
                             albedo_color=particle_color, model="phong")
    rgb = torch.from_numpy(rgb_np).to(x_t.device)

    pred = renderer.render(x_t, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=True)
    pred_alpha = pred.get('alpha')
    if pred_alpha is None:
        return None, None, 0.0
    if pred_alpha.dim() == 3:
        pred_alpha = pred_alpha[0]

    tgt = target_alpha.to(pred_alpha.device)
    if tgt.shape != pred_alpha.shape:
        tgt = F_nn.interpolate(tgt.unsqueeze(0).unsqueeze(0),
                               size=pred_alpha.shape, mode='bilinear',
                               align_corners=False)[0, 0]

    loss = torch.mean((pred_alpha - tgt) ** 2)
    loss.backward()

    dLdx = x_t.grad.detach().cpu().numpy().astype(np.float32) if x_t.grad is not None else np.zeros_like(x)
    dLdF = F_t.grad.detach().cpu().numpy().astype(np.float32) if F_t.grad is not None else np.zeros((x.shape[0], 3, 3), dtype=np.float32)
    return dLdx, dLdF, loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    import diffmpm_bindings
    opt = build_opt_input(cfg)
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    cam_cfg = cfg.get('camera', {})
    render_cfg = cfg.get('render', {})
    particle_color = render_cfg.get('particle_color', [0.27, 0.51, 0.71])
    renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=True)
    campos = view_params.get('campos')

    rs = default_cfg()
    rs.update(adapt_config({'upsample': cfg.get('upsample', {})}))

    target_positions = np.array(target_pc.get_positions(), dtype=np.float32)
    target_alpha = generate_target_render(target_positions, rs, renderer, campos, render_cfg, particle_color)
    if target_alpha is None:
        print("ERROR: target alpha failed")
        return

    cov_cfg = cfg.get('upsample', {}).get('covariance', {})
    sigma0 = float(cov_cfg.get('sigma0', 0.2))
    sv_min = float(cov_cfg.get('sv_min', 0.3))
    num_timesteps = int(opt.num_timesteps)

    # ══════════════════════════════════════════════════════════════
    # PASS 1: Physics only
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PASS 1: Physics only")
    print("=" * 70)
    cg.run_optimization(opt)
    loss1 = cg.end_layer_mass_loss()
    print(f"Physics loss after pass 1: {loss1:.2f}")

    # Extract state
    pc = cg.get_point_cloud(num_timesteps - 1)
    x = np.ascontiguousarray(np.array(pc.get_positions(), dtype=np.float32))
    try:
        F_e = pc.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy()
    except Exception:
        F_e = np.array(pc.get_def_grads(), dtype=np.float32)
    F_e = np.ascontiguousarray(F_e.astype(np.float32))
    N = x.shape[0]

    # Read F_total at control layer (layer 0) after pass 1
    # F_total = F + dFc; since F at layer 0 is constant, ΔF_total = ΔdFc
    pc0_after_pass1 = cg.get_point_cloud(0)
    Ftotal_pass1 = pc0_after_pass1.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy().astype(np.float32)
    # Also read physics-only gradient norms at each layer
    phys_grad_norms = []
    for layer_i in range(num_timesteps):
        try:
            gn = cg.get_layer_phys_grad_norm(layer_i)
            phys_grad_norms.append(gn)
        except Exception:
            break
    if phys_grad_norms:
        print(f"Physics grad norms by layer: first={phys_grad_norms[0]}, last={phys_grad_norms[-1]}")
    print(f"F_total norm at layer 0 after pass 1: {np.linalg.norm(Ftotal_pass1):.6f}")

    # ══════════════════════════════════════════════════════════════
    # Compute render gradients
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Computing render gradients...")
    print("=" * 70)
    dLdx_render, dLdF_render, render_loss = compute_render_grads(
        x, F_e, target_alpha, renderer, campos, render_cfg, particle_color,
        sigma0=sigma0, sv_min=sv_min)

    if dLdF_render is None:
        print("ERROR: render grads failed")
        return

    dLdx_norm = np.linalg.norm(dLdx_render)
    dLdF_norm = np.linalg.norm(dLdF_render)
    print(f"render_loss = {render_loss:.6f}")
    print(f"||dL/dx_render|| = {dLdx_norm:.6e}")
    print(f"||dL/dF_render|| = {dLdF_norm:.6e}")

    # ══════════════════════════════════════════════════════════════
    # PASS 2: Inject render grads + re-run
    # ══════════════════════════════════════════════════════════════
    for gain in [1.0, 10.0, 100.0]:
        print(f"\n{'=' * 70}")
        print(f"PASS 2: render_gain = {gain}")
        print(f"{'=' * 70}")

        cg.set_render_gain(gain)
        cg.set_render_gradients(dLdF_render, dLdx_render)
        cg.run_optimization(opt, skip_setup=True)
        cg.clear_render_gradients()

        loss2 = cg.end_layer_mass_loss()
        print(f"Physics loss after pass 2: {loss2:.2f}")

        pc0_after_pass2 = cg.get_point_cloud(0)
        Ftotal_pass2 = pc0_after_pass2.get_def_grads_total_torch(requires_grad=False).detach().cpu().numpy().astype(np.float32)

        dFc_delta = Ftotal_pass2 - Ftotal_pass1
        dFc_delta_norm = np.linalg.norm(dFc_delta)

        print(f"F_total norm at layer 0 after pass 2: {np.linalg.norm(Ftotal_pass2):.6f}")
        print(f"||ΔF_total|| (≈ΔdFc, pass2 - pass1): {dFc_delta_norm:.6f}")
        print(f"Effective amplification: ||ΔF_total|| / (gain × ||dL/dF_render||) = "
              f"{dFc_delta_norm / (gain * dLdF_norm + 1e-12):.4f}")

        # Per-layer gradient norms for this pass
        render_grad_norms = []
        for layer_i in range(num_timesteps):
            try:
                gn = cg.get_layer_phys_grad_norm(layer_i)
                render_grad_norms.append(gn)
            except Exception:
                break
        if render_grad_norms:
            print(f"Grad norms: first={render_grad_norms[0]}, last={render_grad_norms[-1]}")

        # Per-particle analysis
        dFc_delta_pp = np.linalg.norm(dFc_delta.reshape(N, -1), axis=1)
        print(f"Per-particle ΔF_total: mean={dFc_delta_pp.mean():.6e}, "
              f"max={dFc_delta_pp.max():.6e}, "
              f"non-zero(>1e-10)={np.sum(dFc_delta_pp > 1e-10)}/{N}")

    print("\nDone.")


if __name__ == '__main__':
    main()
