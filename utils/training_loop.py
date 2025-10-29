"""
Training Loop - E2E Episode Training

Main E2E training loop with multi-pass refinement.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

from utils.physics_utils import run_physics_optimization
from utils.rendering_utils import compute_render_loss_pass, extract_render_gradients
from utils.visualization_utils import visualize_episode


# ============================================================================
# E2E Training Episode
# ============================================================================

def run_e2e_episode(
    ep: int,
    cg: Any,
    opt: Any,
    num_timesteps: int,
    control_stride: int,
    num_passes: int,
    rs_full: Dict,
    ema_state: Dict,
    renderer: Any,
    loss_manager: Any,
    target_render: Dict,
    view_params: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    out_dir: Path,
    png_enabled: bool,
    tgt: np.ndarray,
    cov_module=None,
    cov_optimizer=None
) -> Tuple[Dict, Dict]:
    """
    Run one complete E2E training episode with multi-pass refinement.
    
    ════════════════════════════════════════════════════════════════════════════
    E2E (End-to-End) Training Architecture
    ════════════════════════════════════════════════════════════════════════════
    
    This function implements the core training loop that jointly optimizes:
      1. Physics simulation (MPM)
      2. Surface synthesis (Upsampling)
      3. Rendering (3D Gaussian Splatting)
    
    Training Flow (Multi-Pass Refinement):
    ────────────────────────────────────────────────────────────────────────────
    For pass in [1, 2, 3]:
      
      Phase 1: Inject Render Gradients (if pass > 1)
        • Take ∂L_render/∂F, ∂L_render/∂x from previous pass
        • Inject into C++ MPM backend
        • L_total = L_physics + L_render
      
      Phase 2: Physics Optimization
        • Run forward simulation: x(0) → x(1) → ... → x(T)
        • Compute physics loss: L_physics = ||x(T) - x_target||²
        • Backward: compute gradients ∂L_total/∂controls
        • Update: Adam step on control forces
      
      Phase 3: Render Loss Computation
        • Get final state: x_final, F_final
        • Upsample: (x, F) → (μ, Σ, normals) [400k points]
        • Compute shading: RGB from normals + lighting
        • Render: (μ, Σ, RGB) → {image, alpha, depth}
        • Compare with target: L_render = f(pred, target)
        • Backward: L_render.backward() → ∂L/∂F, ∂L/∂x
        • Store gradients for next pass
      
      Phase 4: Visualization (last pass only)
        • Save rendered images (PNG)
        • Export Gaussians (NPZ)
        • Export point cloud (PLY)
        • Generate comparison images
    
    ════════════════════════════════════════════════════════════════════════════
    """
    # ════════════════════════════════════════════════════════════════════════════
    # Episode Initialization
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Episode {ep+1} START")
    print(f"{'='*70}")
    
    # Reset EMA threshold at episode boundary
    if ep > 0 and "ema_thr" in ema_state:
        print(f"\n[Episode Boundary] Resetting EMA threshold")
        ema_state["ema_thr"] = None
    
    # Setup computation graph
    print(f"\n[Setup] Creating {num_timesteps} timestep layers...")
    cg.set_up_comp_graph(num_timesteps)
    
    print(f"[Setup] Running initial forward simulation...")
    cg.compute_forward_pass(0, ep)
    
    try:
        loss_initial = cg.end_layer_mass_loss()
        print(f"[Setup] Initial physics loss: {loss_initial:.2f}")
    except Exception as e:
        print(f"[Setup] Loss computation failed: {e}")
        loss_initial = 0.0
    
    print(f"\n{'='*70}")
    print(f"E2E Training - {num_passes} Passes")
    print(f"{'='*70}")
    
    accumulated_render_grads = None
    final_loss_components = None
    
    # ════════════════════════════════════════════════════════════════════════════
    # Multi-Pass Refinement Loop
    # ════════════════════════════════════════════════════════════════════════════
    for pass_idx in range(num_passes):
        print(f"\n{'─'*70}")
        print(f"Pass {pass_idx+1}/{num_passes}")
        print(f"{'─'*70}")
        
        torch.cuda.empty_cache()
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 1: Inject Render Gradients
        # ────────────────────────────────────────────────────────────────────────
        if accumulated_render_grads is not None:
            dLdF = accumulated_render_grads['dLdF']
            dLdx = accumulated_render_grads['dLdx']
            
            grad_F_norm = np.linalg.norm(dLdF)
            grad_x_norm = np.linalg.norm(dLdx)
            
            print(f"\n[Inject] Applying render gradients from Pass {pass_idx}")
            print(f"├─ Points: {len(dLdF)}")
            print(f"├─ ||∂L_render/∂F|| = {grad_F_norm:.6e}")
            print(f"└─ ||∂L_render/∂x|| = {grad_x_norm:.6e}")
            
            try:
                cg.set_render_gradients(dLdF, dLdx)
                print(f"   ✅ Gradients injected successfully\n")
            except Exception as e:
                print(f"   ❌ Gradient injection failed: {e}\n")
        else:
            print(f"\n[Inject] No previous render grads (first pass)\n")
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 2: Physics Optimization
        # ────────────────────────────────────────────────────────────────────────
        loss_physics = run_physics_optimization(
            cg, opt, num_timesteps, control_stride, ep, pass_idx
        )
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 3: Compute Render Loss
        # ────────────────────────────────────────────────────────────────────────
        seed = 9999 + ep*1000 + pass_idx
        
        print(f"[Render] Computing loss for Pass {pass_idx+1}...")
        
        result = compute_render_loss_pass(
            cg, num_timesteps, rs_full, ema_state, renderer,
            loss_manager, target_render, view_params, campos,
            render_cfg, particle_color, seed, cov_module
        )
        
        if result[0] is not None:
            ema_state_new, F, x, loss_components = result
            
            # Store loss components from last pass
            if pass_idx == num_passes - 1:
                final_loss_components = loss_components
            
            # Update learnable covariance
            if cov_optimizer is not None and cov_module is not None:
                cov_optimizer.step()
                cov_optimizer.zero_grad()
            
            # Extract gradients
            render_grads = extract_render_gradients(F, x)
            if render_grads is not None:
                accumulated_render_grads = render_grads
                print(f"├─ ✅ Render grads saved for Pass {pass_idx+2}")
            
            # Delete tensors
            del F, x
            
            # Update ema_state
            ema_state = ema_state_new
        else:
            print(f"└─ ⚠️ compute_render_loss_pass returned None\n")
        
        # Cleanup
        del result
        torch.cuda.empty_cache()
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 4: Visualization (last pass only)
        # ────────────────────────────────────────────────────────────────────────
        if pass_idx == num_passes - 1:
            print(f"[Visualization] Saving final results...")
            seed = 9999 + ep*1000 + pass_idx
            visualize_episode(
                ep, out_dir, cg, num_timesteps, rs_full, ema_state,
                renderer, campos, render_cfg, particle_color,
                png_enabled, tgt, loss_physics, seed, cov_module
            )
    
    # ════════════════════════════════════════════════════════════════════════════
    # Episode Finalization
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Episode {ep+1} COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"[Cleanup] Final memory cleanup...")
    accumulated_render_grads = None
    
    if cg.has_render_gradients():
        cg.clear_render_gradients()
    
    torch.cuda.empty_cache()
    
    # Prepare final losses
    final_losses = {
        'loss_physics': loss_physics if 'loss_physics' in locals() else 0.0
    }
    
    if final_loss_components is not None:
        final_losses.update(final_loss_components)
    
    return ema_state, final_losses


__all__ = [
    'run_e2e_episode',
]

