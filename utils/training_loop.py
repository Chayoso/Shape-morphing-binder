"""
Training Loop - E2E Episode Training

Main E2E training loop with multi-pass refinement.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

from utils.physics_utils import run_physics_optimization, run_physics_optimization_batched, extract_point_cloud_state
from utils.rendering_utils import compute_render_loss_pass, extract_render_gradients
from utils.visualization_utils import visualize_episode
from utils.gradient_utils import (
    compute_gradient_statistics,
    compute_gradient_cosine_similarity,
    pcgrad_projection,
    ema_gradient_scaling,
    adaptive_target_ratio_schedule,
    diagnose_gradient_health
)


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
    cov_optimizer=None,
    external_levelset=None
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
        • 🔥 Advect level set using final velocity (for next pass)
      
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
        
        # 🔥 Physics weight scheduling (progressive annealing)
        total_animations = rs_full.get('optimization', {}).get('num_animations', 50)
        progress = (ep + pass_idx / num_passes) / max(1, total_animations)
        phys_w = float(np.interp(progress, [0.0, 0.3, 0.7, 1.0], [1.5, 1.2, 1.0, 0.8]))
        
        try:
            cg.set_physics_weight(phys_w)
            print(f"[Physics Weight] progress={progress:.2%}, weight={phys_w:.2f} "
                  f"{'(physics-dominant)' if phys_w > 1.1 else '(balanced)' if phys_w > 0.9 else '(render-refinement)'}")
        except Exception as e:
            print(f"[Physics Weight] Failed: {e}")
        
        torch.cuda.empty_cache()
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 1+2: Physics Optimization with Batched Gradient Injection (OPTIMIZED)
        # ────────────────────────────────────────────────────────────────────────
        # 🔥 NEW: Use batched E2E pass (combines gradient injection + physics optimization)
        use_batched = True  # Set to False to use old method

        if use_batched:
            # Prepare render gradients (if available)
            render_grads_dict = None
            if accumulated_render_grads is not None and renderer is not None and loss_manager is not None:
                render_grads_dict = {
                    'dLdF': accumulated_render_grads['dLdF'],
                    'dLdx': accumulated_render_grads['dLdx']
                }
                grad_F_norm = np.linalg.norm(render_grads_dict['dLdF'])
                grad_x_norm = np.linalg.norm(render_grads_dict['dLdx'])
                print(f"\n[Batched E2E] Pass {pass_idx+1} with render gradients")
                print(f"├─ Points: {len(render_grads_dict['dLdF'])}")
                print(f"├─ ||∂L_render/∂F|| = {grad_F_norm:.6e}")
                print(f"└─ ||∂L_render/∂x|| = {grad_x_norm:.6e}")
            else:
                if pass_idx == 0:
                    print(f"\n[Batched E2E] Pass {pass_idx+1} - Physics-only (first pass)")
                elif renderer is None or loss_manager is None:
                    print(f"\n[Batched E2E] Pass {pass_idx+1} - Physics-only mode")
                else:
                    print(f"\n[Batched E2E] Pass {pass_idx+1} - No render grads available")

            # 🔥 Single batched call (2-3x faster than separate calls!)
            loss_physics = run_physics_optimization_batched(
                cg, opt, render_grads_dict, pass_idx
            )

        else:
            # Old method (kept for fallback/debugging)
            if accumulated_render_grads is not None and renderer is not None and loss_manager is not None:
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
                if pass_idx == 0:
                    print(f"\n[Inject] No previous render grads (first pass)\n")
                elif renderer is None or loss_manager is None:
                    print(f"\n[Inject] Skipped - Physics-only mode\n")
                else:
                    print(f"\n[Inject] No render grads available\n")

            loss_physics = run_physics_optimization(
                cg, opt, num_timesteps, control_stride, ep, pass_idx
            )
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 3: Compute Render Loss
        # ────────────────────────────────────────────────────────────────────────
        seed = 9999 + ep*1000 + pass_idx
        
        print(f"\n[Render] Computing loss for Pass {pass_idx+1}...")
        
        result = compute_render_loss_pass(
            cg, num_timesteps, rs_full, ema_state, renderer,
            loss_manager, target_render, view_params, campos,
            render_cfg, particle_color, seed, cov_module,
            external_levelset=None,
            current_episode=ep
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
            
            # Extract gradients (E2E mode only)
            render_grads = extract_render_gradients(F, x)
            if render_grads is not None and renderer is not None and loss_manager is not None:
                dLdF_render = render_grads['dLdF']
                dLdx_render = render_grads['dLdx']
                
                # ═══════════════════════════════════════════════════════════════
                # Multi-Task Gradient Management
                # ═══════════════════════════════════════════════════════════════
                
                # 1. Diagnose gradient health
                grad_health = diagnose_gradient_health(dLdF_render, dLdx_render, grad_type="render")
                if not grad_health['is_healthy']:
                    print(f"├─ ⚠️  Skipping unhealthy render gradients")
                    accumulated_render_grads = None
                    continue
                
                # 2. Compute gradient statistics
                render_stats = compute_gradient_statistics(dLdF_render, dLdx_render)
                g_render = render_stats['grad_total_norm']
                
                try:
                    # Get physics gradients for comparison
                    gF_phys, gx_phys = cg.get_last_layer_phys_grad_norm()
                    g_phys = max(np.sqrt(gF_phys**2 + gx_phys**2), 1e-12)
                    
                    # Also get full physics gradients for PCGrad (if available)
                    try:
                        dLdF_phys, dLdx_phys = cg.get_last_layer_phys_gradients()
                        has_phys_grads = True
                    except:
                        dLdF_phys = np.zeros_like(dLdF_render)
                        dLdx_phys = np.zeros_like(dLdx_render)
                        has_phys_grads = False
                    
                    # 3. Compute cosine similarity (gradient conflict detection)
                    if has_phys_grads:
                        cosine = compute_gradient_cosine_similarity(
                            dLdF_phys, dLdx_phys, dLdF_render, dLdx_render
                        )
                    else:
                        # Fallback: estimate from norms only
                        dLdF_phys_flat = dLdF_render.flatten()  # Dummy
                        dLdF_render_flat = dLdF_render.flatten()
                        dot = np.dot(dLdF_phys_flat, dLdF_render_flat)
                        cosine = dot / (g_phys * g_render + 1e-12)
                    
                    conflict_status = (
                        '⚠️ CONFLICT' if cosine < -0.3 else
                        '✓ aligned' if cosine > 0.3 else
                        '~ neutral'
                    )
                    
                    # 4. Adaptive target ratio schedule
                    total_animations = rs_full.get('optimization', {}).get('num_animations', 50)
                    target_ratio = adaptive_target_ratio_schedule(
                        episode=ep,
                        total_episodes=total_animations,
                        warmup_episodes=5,
                        rampup_episodes=15,
                        warmup_ratio=0.05,
                        rampup_ratio=0.30,
                        full_ratio=0.50
                    )
                    
                    # 5. EMA-based gradient scaling
                    # 🔥 초기 에피소드에서는 EMA를 덜 사용 (빠른 적응)
                    ema_beta_adaptive = 0.5 if ep < 10 else 0.8
                    
                    # 🔥 초기에는 더 공격적인 power (빠른 수렴)
                    power_adaptive = 0.85 if ep < 10 else 0.75
                    
                    gain, scaling_info = ema_gradient_scaling(
                        g_render_norm=g_render,
                        g_physics_norm=g_phys,
                        target_ratio=target_ratio,
                        ema_state=ema_state,
                        ema_beta=ema_beta_adaptive,  # 초기: 0.5 (빠른 반응), 후기: 0.8 (안정)
                        power=power_adaptive,        # 초기: 0.85 (공격적), 후기: 0.75 (보수적)
                        min_gain=0.1,
                        max_gain=50000.0  # 🔥 극단적 불균형 대응
                    )
                    
                    # 6. PCGrad projection (if conflict detected)
                    pcgrad_enabled = rs_full.get('optimization', {}).get('use_pcgrad', True)
                    if pcgrad_enabled and has_phys_grads and cosine < -0.1:
                        dLdF_render, dLdx_render, pcgrad_info = pcgrad_projection(
                            dLdF_render=dLdF_render,
                            dLdx_render=dLdx_render,
                            dLdF_physics=dLdF_phys,
                            dLdx_physics=dLdx_phys,
                            conflict_threshold=-0.1
                        )
                        pcgrad_applied = pcgrad_info['pcgrad_applied']
                        pcgrad_scale = pcgrad_info['pcgrad_projection_scale']
                    else:
                        pcgrad_applied = False
                        pcgrad_scale = 0.0
                    
                    # 7. Apply gain
                    dLdF_render *= gain
                    dLdx_render *= gain
                    
                    # Update render_grads
                    render_grads['dLdF'] = dLdF_render
                    render_grads['dLdx'] = dLdx_render
                    
                    # ══════════════════════════════════════════════════════════
                    # Diagnostic Logging
                    # ══════════════════════════════════════════════════════════
                    print(f"\n├─ [Gradient Diagnostics] Pass {pass_idx+1}")
                    print(f"│  ├─ Norms: ||g_render||={g_render:.3e}, ||g_phys||={g_phys:.3e}")
                    print(f"│  ├─ Ratio: actual={g_render/g_phys:.6f}, target={target_ratio:.4f}")
                    
                    # 🔥 Gain 클램프 경고
                    is_clamped_max = (scaling_info['gain_raw'] >= 10000.0)
                    is_clamped_min = (scaling_info['gain_raw'] <= 0.1)
                    clamp_warning = ""
                    if is_clamped_max:
                        clamp_warning = " ⚠️ CLAMPED AT MAX!"
                    elif is_clamped_min:
                        clamp_warning = " ⚠️ CLAMPED AT MIN!"
                    
                    print(f"│  ├─ Gain: raw={scaling_info['gain_raw']:.2f}, smoothed={gain:.2f}{clamp_warning}")
                    print(f"│  │  ├─ EMA beta={ema_beta_adaptive:.2f}, power={power_adaptive:.2f}")
                    
                    # 🔥 after_gain_ratio 출력 (핵심 지표!)
                    after_gain_ratio = (g_render * gain) / g_phys
                    print(f"│  │  └─ Ratio after gain: {after_gain_ratio:.6f} (target: {target_ratio:.4f})")
                    
                    # 🔥 목표 대비 달성도
                    ratio_achievement = after_gain_ratio / target_ratio if target_ratio > 0 else 0.0
                    achievement_str = f"{ratio_achievement*100:.1f}% of target"
                    if ratio_achievement < 0.5:
                        achievement_str += " ⚠️ TOO LOW"
                    elif ratio_achievement > 1.5:
                        achievement_str += " ⚠️ TOO HIGH"
                    print(f"│  │     ({achievement_str})")
                    
                    print(f"│  ├─ Conflict: cos(θ)={cosine:+.4f} {conflict_status}")
                    if pcgrad_applied:
                        print(f"│  ├─ PCGrad: ✅ Applied (projection scale={pcgrad_scale:.2%})")
                    else:
                        print(f"│  ├─ PCGrad: ⊘ Not needed (cos(θ) ≥ -0.1)")
                    print(f"│  └─ Target schedule: ep={ep} → ratio={target_ratio:.3f}")
                    
                except Exception as e:
                    print(f"├─ [Gradient Management] Error: {e}")
                    print(f"├─ [Fallback] Using default gain=1.0")
                    gain = 1.0
                
                accumulated_render_grads = render_grads
                print(f"└─ ✅ Render grads processed and saved for Pass {pass_idx+2}\n")
            else:
                if renderer is None or loss_manager is None:
                    print(f"├─ ⚠️  Physics-only mode - no render grads")
            
            del F, x
            ema_state = ema_state_new
        else:
            print(f"└─ ⚠️ compute_render_loss_pass returned None\n")
        
        del result
        torch.cuda.empty_cache()
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 4: Visualization (last pass only)
        # ────────────────────────────────────────────────────────────────────────
        if pass_idx == num_passes - 1:
            print(f"\n[Visualization] Saving final results...")
            seed = 9999 + ep*1000 + pass_idx
            visualize_episode(
                ep, out_dir, cg, num_timesteps, rs_full, ema_state,
                renderer, campos, render_cfg, particle_color,
                png_enabled, tgt, loss_physics, seed, cov_module,
                external_levelset=external_levelset
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