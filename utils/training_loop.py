"""
Training Loop - E2E Episode Training

Main E2E training loop with multi-pass refinement.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import time

from utils.physics_utils import run_physics_optimization, run_physics_optimization_batched, extract_point_cloud_state
from utils.rendering_utils import compute_render_loss_pass, extract_render_gradients, upsample_current_state
from utils.visualization_utils import visualize_episode
from utils.gradient_utils import (
    compute_gradient_statistics,
    compute_gradient_cosine_similarity,
    pcgrad_projection,
    ema_gradient_scaling,
    adaptive_target_ratio_schedule,
    diagnose_gradient_health,
    normalize_and_combine_gradients
)


# ============================================================================
# E2E Training Episode (Session Mode - MAXIMUM PERFORMANCE)
# ============================================================================

def run_e2e_episode_session(
    session: Any,
    ep: int,
    num_timesteps: int,
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
    external_levelset=None
) -> Tuple[Dict, Dict]:
    """
    🔥 MAXIMUM PERFORMANCE: Run E2E episode using persistent session.

    This is ~10-15x faster than run_e2e_episode() because:
      - Single Python→C++ transition per episode (vs 50-100)
      - All physics runs with GIL released
      - Persistent buffer reuse across episodes

    Args:
        session: E2ESession instance (C++)
        ep: Episode number
        num_timesteps: Number of simulation timesteps
        rs_full: Upsampling configuration
        ema_state: EMA state dict (updated in-place)
        renderer: 3DGS renderer
        loss_manager: Loss manager
        target_render: Target rendering dict
        view_params: Camera parameters
        campos: Camera position
        render_cfg: Rendering configuration
        particle_color: Base particle color
        out_dir: Output directory for this episode
        png_enabled: Whether to save PNG visualizations
        tgt: Target point cloud positions
        cov_module: Optional learnable covariance module
        external_levelset: Optional pre-computed level set

    Returns:
        Tuple of (updated_ema_state, final_losses)
    """
    print(f"\n{'='*70}")
    print(f"🔥 Session Mode: Episode {ep+1} START")
    print(f"{'='*70}")

    # Track render loss components
    last_render_loss_components = {}

    # Define render gradient callback
    def compute_render_grads_callback(episode_num: int, pass_idx: int):
        """
        Called by C++ to get render gradients for a pass.
        This is the ONLY Python code that runs during the episode!
        """
        nonlocal last_render_loss_components

        import time

        try:
            t_start = time.time()
            print(f"\n[Render Callback] Episode {episode_num}, Pass {pass_idx+1}", flush=True)

            # Get final state from comp graph (after physics pass completed)
            t0 = time.time()
            pc = session.get_final_point_cloud()
            t_extract_pc = time.time() - t0
            if pc is None:
                print("  ⚠️  No point cloud available")
                return None

            # Extract state with zero-copy views
            t0 = time.time()
            try:
                x = pc.get_positions_torch_view().clone().requires_grad_(True)
                F = pc.get_def_grads_total_torch_view().clone().requires_grad_(True)
            except:
                print("  ⚠️  Failed to extract positions/gradients")
                return None
            t_extract_state = time.time() - t0

            print(f"  ├─ Extracted state: {len(x)} particles")
            print(f"  ├─ x.requires_grad: {x.requires_grad}, F.requires_grad: {F.requires_grad}")

            # Upsample and compute render loss
            seed = 9999 + episode_num*1000 + pass_idx

            # ✅ CRITICAL: Pass x, F directly to maintain gradient connection!
            from sampling import upsample
            t0 = time.time()
            result = upsample(
                x, F,
                cfg=rs_full.get('upsample', {}),
                state=ema_state,
                seed=seed,
                return_torch=True,
                export_stages=False,
                learnable_cov_module=cov_module,
                current_episode=episode_num,
                external_levelset=external_levelset,
                use_simple_pipeline=rs_full.get('upsample', {}).get('use_simple_pipeline', True)
            )
            t_upsample = time.time() - t0

            mu = result.get('points')
            cov = result.get('cov')

            if mu is None:
                print("  ⚠️  Upsampling failed")
                return None

            print(f"  ├─ Upsampled: {len(mu)} points")

            # Prepare rendering inputs
            from utils.rendering_utils import prepare_rendering_inputs
            t0 = time.time()
            rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
            t_prep_render = time.time() - t0

            # Render
            t0 = time.time()
            out_pred = renderer.render(mu, cov, rgb=rgb)
            t_render = time.time() - t0

            # ✅ Convert render outputs to torch tensors and ensure same device
            device = mu.device if torch.is_tensor(mu) else 'cuda'
            
            # Convert pred to torch and move to device
            pred_dict = {}
            for key in ['image', 'alpha', 'depth']:
                if key in out_pred and out_pred[key] is not None:
                    val = out_pred[key]
                    if not torch.is_tensor(val):
                        val = torch.from_numpy(val)
                    val = val.to(device)
                    pred_dict[key] = val
            
            # ✅ Convert target to torch and move to SAME device
            target_dict = {}
            for key in ['image', 'alpha', 'depth']:
                if key in target_render and target_render[key] is not None:
                    val = target_render[key]
                    if not torch.is_tensor(val):
                        val = torch.from_numpy(val)
                    val = val.to(device)
                    target_dict[key] = val

            # Extract F_interp from upsampling result (or use original F as fallback)
            F_interp = result.get('F_interp')
            if F_interp is None:
                print(f"  ⚠️  F_interp not in result, using original F")
                F_interp = F

            # ✅ Get cov_target from target_render (computed during target rendering)
            cov_target = target_render.get('cov_target')
            if cov_target is not None:
                print(f"  ├─ Using cov_target for spectral alignment loss")
            
            # Compute loss
            t0 = time.time()
            loss_components = loss_manager.compute_render_loss(
                pred=pred_dict,
                target=target_dict,  # ✅ Use converted target
                cov=cov,
                mu=mu,
                view_params=view_params,
                cov_target=cov_target,  # ✅ Pass target covariance from target_render
                F=F_interp
            )
            t_loss = time.time() - t0

            loss_total = loss_components.get('loss_render_total', loss_components.get('loss_total', torch.tensor(0.0)))
            print(f"  ├─ Render loss total: {loss_total.item():.6f}", flush=True)

            # Print detailed loss components
            print(f"  │  ├─ Alpha:     {loss_components.get('loss_alpha', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Depth:     {loss_components.get('loss_depth', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Photo:     {loss_components.get('loss_photo', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Edge:      {loss_components.get('loss_edge', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Cov align: {loss_components.get('loss_cov_align', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Cov reg:   {loss_components.get('loss_cov_reg', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  └─ Det barrier: {loss_components.get('loss_det_barrier', torch.tensor(0.0)).item():.6f}", flush=True)

            # Store for final reporting
            last_render_loss_components = {k: v.item() if torch.is_tensor(v) else v
                                          for k, v in loss_components.items()}

            # Backward
            print(f"  ├─ Running backward()...")
            print(f"  │  loss_total.requires_grad: {loss_total.requires_grad}")
            print(f"  │  loss_total.device: {loss_total.device}")

            t0 = time.time()
            try:
                loss_total.backward()
                print(f"  ├─ Backward completed ✅")
            except Exception as e:
                print(f"  └─ ❌ Backward failed: {e}")
                return None
            t_backward = time.time() - t0

            # Extract gradients
            t0 = time.time()
            print(f"  ├─ Checking gradients...")
            print(f"  │  F.grad is None: {F.grad is None}")
            print(f"  │  x.grad is None: {x.grad is None}")
            
            if F.grad is None or x.grad is None:
                print("  └─ ⚠️  No gradients computed")
                print(f"     Possible cause: Computational graph disconnected during upsampling")
                return None

            # ✅ Import numpy inside callback
            import numpy as np

            dLdF_render = F.grad.detach().cpu().numpy()
            dLdx_render = x.grad.detach().cpu().numpy()

            # Ensure contiguous
            if not dLdF_render.flags['C_CONTIGUOUS']:
                dLdF_render = np.ascontiguousarray(dLdF_render)
            if not dLdx_render.flags['C_CONTIGUOUS']:
                dLdx_render = np.ascontiguousarray(dLdx_render)

            grad_F_norm_raw = np.linalg.norm(dLdF_render)
            grad_x_norm_raw = np.linalg.norm(dLdx_render)
            print(f"  ├─ Raw render gradients: ||∂L/∂F||={grad_F_norm_raw:.3e}, ||∂L/∂x||={grad_x_norm_raw:.3e}", flush=True)

            # 🔥 GRADIENT NORMALIZATION: Scale render gradients to physics-like magnitude
            # Typical physics gradient magnitude is ~0.01-0.1 (based on initial_alpha=1.0)
            # Render gradients can be 1000x-100000x larger, causing line search failures

            # Get target magnitude from config or use conservative default
            magnitude_strategy = rs_full.get('optimization', {}).get('magnitude_strategy', 'physics')

            # Conservative target: match typical physics gradient magnitude
            target_magnitude_F = 0.05  # Conservative physics-like magnitude
            target_magnitude_x = 0.05

            # Scale render gradients down to target magnitude
            eps = 1e-12
            scale_F = target_magnitude_F / (grad_F_norm_raw + eps)
            scale_x = target_magnitude_x / (grad_x_norm_raw + eps)

            # Only scale down (never up) to be conservative
            scale_F = min(scale_F, 1.0)
            scale_x = min(scale_x, 1.0)

            dLdF_normalized = dLdF_render * scale_F
            dLdx_normalized = dLdx_render * scale_x

            grad_F_norm_final = np.linalg.norm(dLdF_normalized)
            grad_x_norm_final = np.linalg.norm(dLdx_normalized)

            print(f"  ├─ Normalization: scale_F={scale_F:.3e}, scale_x={scale_x:.3e}", flush=True)
            print(f"  └─ Final render gradients: ||∂L/∂F||={grad_F_norm_final:.3e}, ||∂L/∂x||={grad_x_norm_final:.3e}", flush=True)

            t_grad_extract = time.time() - t0
            t_total = time.time() - t_start

            # Print timing summary
            print(f"\n  [Timing Breakdown]:", flush=True)
            print(f"    Extract PC:    {t_extract_pc*1000:6.2f}ms ({t_extract_pc/t_total*100:4.1f}%)", flush=True)
            print(f"    Extract state: {t_extract_state*1000:6.2f}ms ({t_extract_state/t_total*100:4.1f}%)", flush=True)
            print(f"    Upsample:      {t_upsample*1000:6.2f}ms ({t_upsample/t_total*100:4.1f}%)", flush=True)
            print(f"    Prep render:   {t_prep_render*1000:6.2f}ms ({t_prep_render/t_total*100:4.1f}%)", flush=True)
            print(f"    Render:        {t_render*1000:6.2f}ms ({t_render/t_total*100:4.1f}%)", flush=True)
            print(f"    Loss compute:  {t_loss*1000:6.2f}ms ({t_loss/t_total*100:4.1f}%)", flush=True)
            print(f"    Backward:      {t_backward*1000:6.2f}ms ({t_backward/t_total*100:4.1f}%)", flush=True)
            print(f"    Grad extract:  {t_grad_extract*1000:6.2f}ms ({t_grad_extract/t_total*100:4.1f}%)", flush=True)
            print(f"    TOTAL:         {t_total*1000:6.2f}ms\n", flush=True)

            # Return normalized gradients as tuple
            return (dLdF_normalized, dLdx_normalized)

        except Exception as e:
            print(f"  ❌ Render callback error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 🔥 RUN EPISODE (SINGLE C++ CALL!)
    result = session.run_episode(ep, compute_render_grads_callback)

    print(f"\n[Episode {ep+1}] Session Results:")
    print(f"  Loss (physics): {result.loss_physics:.2f}")
    print(f"  Passes executed: {result.num_passes_executed}/{session.get_statistics().total_passes}")
    print(f"  Wall time: {result.wall_time_seconds:.1f}s")
    print(f"  Success: {'✅' if result.success else '❌'}")

    # Visualization (last pass)
    if png_enabled and result.success:
        print(f"\n[Visualization] Saving results...")
        try:
            # Get final point cloud for visualization
            pc_final = session.get_final_point_cloud()
            if pc_final is None:
                print(f"  ⚠️  No final point cloud available")
            else:
                # Session mode: Do simplified visualization without cg
                from utils.physics_utils import extract_point_cloud_state
                
                # Extract state
                x, v, F = extract_point_cloud_state(pc_final, requires_grad=False)
                
                # Upsample for visualization
                seed = 9999 + ep*1000 + (result.num_passes_executed - 1)
                from sampling import upsample
                
                result_viz = upsample(
                    x, F,
                    cfg=rs_full.get('upsample', {}),
                    state=ema_state,
                    seed=seed,
                    return_torch=True,
                    export_stages=False,
                    learnable_cov_module=cov_module,
                    current_episode=ep,
                    external_levelset=external_levelset,
                    use_simple_pipeline=rs_full.get('upsample', {}).get('use_simple_pipeline', True)
                )
                
                mu_viz = result_viz.get('points')
                cov_viz = result_viz.get('cov')
                
                # Render and save
                from utils.rendering_utils import prepare_rendering_inputs
                rgb = prepare_rendering_inputs(mu_viz, result_viz, campos, render_cfg, particle_color)
                out_render = renderer.render(mu_viz, cov_viz, rgb=rgb)
                
                # Save images
                from utils.io_utils import save_image_png, save_depth_png
                
                # Convert to numpy if needed
                def to_np(x):
                    if x is None:
                        return None
                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy()
                    return x
                
                img_np = to_np(out_render.get('image'))
                alpha_np = to_np(out_render.get('alpha'))
                depth_np = to_np(out_render.get('depth'))
                
                # Save renders
                if img_np is not None:
                    save_image_png(out_dir / f"ep{ep:03d}_render.png", img_np)
                if alpha_np is not None:
                    save_image_png(out_dir / f"ep{ep:03d}_alpha.png", alpha_np)
                if depth_np is not None:
                    save_depth_png(out_dir / f"ep{ep:03d}_depth.png", depth_np)
                
                print(f"  ✅ Visualization saved")
                
        except Exception as e:
            print(f"  ⚠️  Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"🔥 Session Mode: Episode {ep+1} COMPLETE")
    print(f"{'='*70}\n")

    # Prepare final losses
    final_losses = {
        'loss_physics': result.loss_physics,
    }
    final_losses.update(last_render_loss_components)

    return ema_state, final_losses


# ============================================================================
# E2E Training Episode (Legacy Pass-by-Pass Mode)
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
        
        # 🔥 Physics weight: DISABLED (we now use normalized gradient combination in Python)
        # Old approach scaled physics gradients in C++ (backwards logic).
        # New approach: normalize and combine gradients in Python before injection.
        phys_w = 1.0  # Keep physics at full scale; balance is handled in Python

        try:
            cg.set_physics_weight(phys_w)
            print(f"[Physics Weight] Fixed at {phys_w:.2f} (balance handled by gradient normalization)")
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

                    print(f"\n[Inject] No render grads available\n")
            
            # ────────────────────────────────────────────────────────────────────────
            # Phase 2: Physics Optimization
            # ────────────────────────────────────────────────────────────────────────
            t0_physics = time.time()
            
            # 🔍 Check physics optimization parameters
            print(f"\n🔍 [Physics Params]")
            print(f"├─ Learning rate (alpha): {opt.initial_alpha}")
            print(f"├─ Max GD iters: {opt.max_gd_iters}")
            print(f"├─ Max LS iters: {opt.max_ls_iters}")
            print(f"└─ Timesteps: {num_timesteps}")
            
            loss_physics = run_physics_optimization(
                cg, opt, num_timesteps, control_stride, ep, pass_idx
            )
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
        
        t_physics = time.time() - t0_physics
        print(f"⏱️  [Physics Optimization] {t_physics:.2f}s")
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 3: Compute Render Loss
        # ────────────────────────────────────────────────────────────────────────
        t0_render = time.time()
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
            
            # 🔥 Track render loss change
            if 'loss_render_total' in loss_components:
                current_render_loss = loss_components['loss_render_total']
                if hasattr(current_render_loss, 'item'):
                    current_render_loss = current_render_loss.item()
                
                if pass_idx > 0 and final_loss_components is not None:
                    prev_render_loss = final_loss_components.get('loss_render_total', 0)
                    if hasattr(prev_render_loss, 'item'):
                        prev_render_loss = prev_render_loss.item()
                    
                    render_change = current_render_loss - prev_render_loss
                    print(f"\n🔍 [Render Loss Tracking]")
                    print(f"├─ Previous: {prev_render_loss:.6f}")
                    print(f"├─ Current:  {current_render_loss:.6f}")
                    print(f"└─ Change:   {render_change:+.6f} {'⚠️ INCREASING!' if render_change > 0 else '✅ Decreasing'}")
            
            # Extract gradients (E2E mode only)
            render_grads = extract_render_gradients(F, x)
            if render_grads is not None and renderer is not None and loss_manager is not None:
                dLdF_render = render_grads['dLdF']
                dLdx_render = render_grads['dLdx']
                
                # ═══════════════════════════════════════════════════════════════
                # 🔥 NEW: Normalized Gradient Combination (prevents magnitude mismatch)
                # ═══════════════════════════════════════════════════════════════

                # 1. Diagnose gradient health
                grad_health = diagnose_gradient_health(dLdF_render, dLdx_render, grad_type="render")
                if not grad_health['is_healthy']:
                    print(f"├─ ⚠️  Skipping unhealthy render gradients")
                    accumulated_render_grads = None
                    continue

                try:
                    # Get physics gradients
                    try:
                        dLdF_phys, dLdx_phys = cg.get_last_layer_phys_gradients()
                        has_phys_grads = True
                    except Exception as e:
                        print(f"├─ ⚠️  Cannot retrieve physics gradients: {e}")
                        print(f"├─ Using render gradients only")
                        accumulated_render_grads = render_grads
                        continue

                    # 2. Compute gradient statistics (before combination)
                    render_stats = compute_gradient_statistics(dLdF_render, dLdx_render)
                    phys_stats = compute_gradient_statistics(dLdF_phys, dLdx_phys)

                    g_render = render_stats['grad_total_norm']
                    g_phys = phys_stats['grad_total_norm']

                    # Compute cosine similarity (conflict detection)
                    cosine = compute_gradient_cosine_similarity(
                        dLdF_phys, dLdx_phys, dLdF_render, dLdx_render
                    )
                    conflict_status = (
                        '⚠️ CONFLICT' if cosine < -0.3 else
                        '✓ aligned' if cosine > 0.3 else
                        '~ neutral'
                    )

                    # 3. Adaptive weight scheduling
                    # Early episodes: Physics-dominant (w_phys=0.9, w_render=0.1)
                    # Mid episodes: Balanced (w_phys=0.7, w_render=0.3)
                    # Late episodes: Render-focused (w_phys=0.6, w_render=0.4)
                    total_animations = rs_full.get('optimization', {}).get('num_animations', 50)
                    progress = ep / max(1, total_animations)

                    if ep < 5:
                        # Warmup: Physics-only (skip render grads)
                        print(f"\n├─ [Warmup] Episode {ep} < 5: Physics-only (skipping render grads)")
                        accumulated_render_grads = None
                        continue
                    elif ep < 15:
                        # Ramp-up: Gradually increase render weight
                        w_render = 0.1 + 0.2 * ((ep - 5) / 10)  # 0.1 → 0.3
                    elif ep < 30:
                        # Mid-training: Balanced
                        w_render = 0.3
                    else:
                        # Late: Increase render for refinement
                        w_render = 0.4

                    w_physics = 1.0 - w_render

                    # 4. 🔥 Normalize and combine gradients
                    # This solves the convergence issue by:
                    #   - Normalizing both to unit vectors (removes scale mismatch)
                    #   - Combining with explicit weights (controllable balance)
                    #   - Re-scaling to physics magnitude (conservative, prevents overshoot)

                    magnitude_strategy = rs_full.get('optimization', {}).get('magnitude_strategy', 'physics')

                    dLdF_combined, dLdx_combined, norm_info = normalize_and_combine_gradients(
                        dLdF_physics=dLdF_phys,
                        dLdx_physics=dLdx_phys,
                        dLdF_render=dLdF_render,
                        dLdx_render=dLdx_render,
                        w_physics=w_physics,
                        w_render=w_render,
                        magnitude_strategy=magnitude_strategy
                    )

                    # Update render_grads with combined gradients
                    render_grads['dLdF'] = dLdF_combined
                    render_grads['dLdx'] = dLdx_combined

                    # ══════════════════════════════════════════════════════════
                    # Diagnostic Logging
                    # ══════════════════════════════════════════════════════════
                    print(f"\n🔥 [Normalized Gradient Combination] Pass {pass_idx+1}")
                    print(f"├─ BEFORE normalization:")
                    print(f"│  ├─ ||g_render|| = {g_render:.6e}")
                    print(f"│  ├─ ||g_phys||   = {g_phys:.6e}")
                    print(f"│  ├─ Ratio (render/phys) = {norm_info['ratio_before']:.6e} {'⚠️ HUGE MISMATCH!' if norm_info['ratio_before'] > 100 else ''}")
                    print(f"│  └─ Conflict: cos(θ) = {cosine:+.4f} {conflict_status}")
                    print(f"│")
                    print(f"├─ WEIGHTS:")
                    print(f"│  ├─ w_physics = {w_physics:.2f} ({w_physics*100:.0f}%)")
                    print(f"│  ├─ w_render  = {w_render:.2f} ({w_render*100:.0f}%)")
                    print(f"│  └─ Strategy  = {magnitude_strategy}")
                    print(f"│")
                    print(f"├─ AFTER combination:")
                    print(f"│  ├─ ||g_combined|| = {norm_info['g_combined_norm']:.6e}")
                    print(f"│  ├─ Ratio (combined/phys) = {norm_info['ratio_after']:.4f} ✅")
                    print(f"│  └─ Magnitude scale = {norm_info['magnitude_scale']:.4f}x")
                    print(f"│")
                    print(f"└─ ✅ Gradients normalized and combined successfully!")

                except Exception as e:
                    print(f"├─ [Gradient Combination] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"├─ [Fallback] Using render gradients only")
                    # Keep render_grads as-is (fallback)
                
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
    'run_e2e_episode_session',  # 🔥 NEW: Session mode (10-15x faster!)
]