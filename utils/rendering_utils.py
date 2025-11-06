"""
Rendering Utilities - 3D Gaussian Splatting & Target Rendering

Handles renderer setup, target rendering, and loss computation.
"""

import numpy as np
import torch
import torch.nn.functional as F_nn  # 🔥 F와 충돌 방지
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from renderer import GSRenderer3DGS, make_matrices_from_yaml, compute_shading
from sampling import upsample


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BG_COLOR = [1.0, 1.0, 1.0]
DEFAULT_PARTICLE_COLOR = [0.27, 0.51, 0.71]


# ============================================================================
# Renderer Setup
# ============================================================================

def setup_renderer(
    cam_cfg: Dict,
    render_cfg: Dict,
    training_mode: bool = False  # 🚀 NEW: Enable resolution downscaling during training
) -> Tuple[Optional[Any], Dict]:
    """
    Initialize 3D Gaussian Splatting renderer with camera configuration.

    Args:
        cam_cfg: Camera configuration (resolution, focal length, pose)
        render_cfg: Rendering configuration (bg color, scale_modifier)
        training_mode: If True, applies resolution downscaling for faster training

    Returns:
        Tuple of (renderer, view_params)
          renderer: GSRenderer3DGS instance or None if failed
          view_params: Camera view matrices and parameters
    """
    try:
        W, H, tanfovx, tanfovy, view_T, proj_T, campos = make_matrices_from_yaml(cam_cfg)

        # 🚀 OPTIMIZATION: Reduce resolution during training (4K→1080p = 4x speedup)
        if training_mode:
            training_scale = render_cfg.get("training_resolution_scale", 0.5)
            if training_scale < 1.0:
                W_train = int(W * training_scale)
                H_train = int(H * training_scale)
                print(f"[Renderer] Training mode: {W}x{H} → {W_train}x{H_train} ({training_scale:.2f}x scale)")
                W, H = W_train, H_train

        bg = render_cfg.get("bg", DEFAULT_BG_COLOR)
        antialiasing = render_cfg.get("antialiasing", False)
        scale_modifier = render_cfg.get("scale_modifier", 1.0)

        renderer = GSRenderer3DGS(
            W, H, tanfovx, tanfovy, view_T, proj_T, campos,
            bg=tuple(bg),
            sh_degree=0,
            scale_modifier=scale_modifier,
            prefiltered=False,
            debug=False,
            antialiasing=antialiasing,
            device="cuda"
        )

        view_params = {
            'view_T': view_T,
            'W': W, 'H': H,
            'tanfovx': tanfovx,
            'tanfovy': tanfovy,
            'campos': campos,
        }

        return renderer, view_params

    except Exception as e:
        print(f"[WARN] 3DGS renderer failed to initialize: {e}")
        return None, {}


# ============================================================================
# Target Rendering
# ============================================================================

def upsample_target(
    x_tgt: np.ndarray, 
    F_tgt: np.ndarray, 
    rs: Dict,
    export_stages: bool = True,
    output_dir: Optional[Path] = None  # 🔥 NEW: output directory for stage exports
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict]]:
    """
    Upsample target point cloud to create dense surface.
    
    🔥 Note: Sets episode=-1 for target mesh (initial frame)
    🔥 NEW: Applies curvature-based covariance (planarity + anisotropy)
    
    Args:
        x_tgt: Target positions (N_target, 3)
        F_tgt: Target deformation gradients (N_target, 3, 3)
        rs: Upsampling configuration
        export_stages: Export intermediate stages
        output_dir: Output directory (for stage exports)
    
    Returns:
        Tuple of (mu_tgt, cov_tgt, nrm_tgt, result_tgt)
    """
    # 🔥 Set episode=-1 for target mesh (initial frame)
    if "covariance" not in rs:
        rs["covariance"] = {}
    rs["covariance"]["episode"] = -1
    
    # 🔥 Set output_dir for stage exports
    if output_dir is not None and "debug" not in rs:
        rs["debug"] = {}
    if output_dir is not None:
        rs["debug"]["output_dir"] = str(output_dir)
    
    result_tgt = upsample(
        x_tgt, F_tgt,
        cfg=rs,               
        seed=9999,
        return_torch=False,
        export_stages=export_stages,
        current_episode=-1  # 🔥 NEW: Mark as target
    )
    
    # NOTE: Curvature-based covariance is now computed directly in STAGE 6 of pipeline
    # when episode <= 0 (target mesh). No post-processing patch needed!
    # See sampling/pipeline.py STAGE 6 for implementation.
    
    mu_tgt = result_tgt["points"]
    cov_tgt = result_tgt.get("cov_target", result_tgt["cov"])  # Try cov_target first, fallback to cov
    nrm_tgt = result_tgt.get("normals")
    
    return mu_tgt, cov_tgt, nrm_tgt, result_tgt


# NOTE: Function removed - target covariance is now computed in STAGE 6
# See sampling/pipeline.py STAGE 6 for implementation
# def compute_target_covariance_star(...): REMOVED


def render_target(
    renderer: Any,
    mu_tgt: np.ndarray,
    cov_tgt: np.ndarray,
    nrm_tgt: Optional[np.ndarray],
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    cov_target_star: Optional[np.ndarray] = None,
    use_curvature_cov: bool = True
) -> Dict:
    """
    Render target point cloud.
    
    Args:
        renderer: Renderer instance
        mu_tgt: (N, 3) target positions
        cov_tgt: (N, 3, 3) F-based covariances
        nrm_tgt: (N, 3) normals
        campos: (3,) camera position
        render_cfg: Render configuration
        particle_color: RGB color
        cov_target_star: (N, 3, 3) curvature-based covariances (optional)
        use_curvature_cov: Use curvature-based cov if available
    
    Returns:
        out_tgt: Render outputs {image, alpha, depth, normal_map}
    """
    if nrm_tgt is None:
        nrm_tgt = np.zeros_like(mu_tgt)
    
    # Choose covariance
    if use_curvature_cov and cov_target_star is not None:
        print("  [Render] Using CURVATURE-based covariance Σ★")
        # Convert torch to numpy if needed
        if hasattr(cov_target_star, 'cpu'):
            cov_to_use = cov_target_star.detach().cpu().numpy()
        else:
            cov_to_use = cov_target_star
    else:
        print("  [Render] Using DEFORMATION GRADIENT-based covariance")
        cov_to_use = cov_tgt
    
    # Statistics
    cov_det = np.linalg.det(cov_to_use)
    print(f"  [Cov Stats] det(Σ): mean={cov_det.mean():.2e}, min={cov_det.min():.2e}, max={cov_det.max():.2e}")
    
    # Compute shading
    rgb_tgt = compute_shading(
        mu_tgt,
        nrm_tgt,
        camera_pos=campos,
        light_cfg=render_cfg.get("lighting", {}),
        albedo_color=particle_color,
        model="phong"
    )
    
    # Render
    out_tgt = renderer.render(
        mu_tgt, cov_to_use, rgb=rgb_tgt,
        normals=nrm_tgt,  
        prefer_cov_precomp=True,
        return_torch=False,
        render_normal_map=True  
    )
    
    return out_tgt


def normalize_render_outputs(out_tgt: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Normalize and extract render outputs.
    
    Args:
        out_tgt: Render output dictionary
    
    Returns:
        Tuple of (img_tgt, alpha_tgt, depth_tgt, normal_map_tgt)
    """
    img_tgt = out_tgt.get('image')
    alpha_tgt = out_tgt.get('alpha')
    depth_tgt = out_tgt.get('depth')
    normal_map_tgt = out_tgt.get('normal_map')
    
    if img_tgt is not None:
        img_tgt = np.clip(img_tgt, 0, 1).astype(np.float32)
    
    if alpha_tgt is not None:
        alpha_tgt = np.clip(alpha_tgt, 0, 1).astype(np.float32)
    
    if depth_tgt is not None:
        depth_tgt = depth_tgt.astype(np.float32)
    
    if normal_map_tgt is not None:
        normal_map_tgt = np.clip(normal_map_tgt, 0, 1).astype(np.float32)
    
    return img_tgt, alpha_tgt, depth_tgt, normal_map_tgt


def create_target_render(
    target_pc: Any,
    renderer: Any,
    rs: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    out_dir: Path
) -> Optional[Dict]:
    """
    Create target rendering from target mesh for E2E supervision.
    
    Pipeline:
      1. Extract target point cloud (x_tgt, F_tgt)
      2. Upsample to dense surface
      3. Compute curvature-based target covariance
      4. Render target image, alpha, depth, normals
      5. Save visualizations to output/target/
    
    Args:
        target_pc: Target point cloud from MPM
        renderer: 3DGS renderer
        rs: Upsampling configuration
        campos: Camera position
        render_cfg: Rendering configuration
        particle_color: RGB color
        out_dir: Output directory
    
    Returns:
        target_render: Dict with {image, alpha, depth, normal_map, cov_target}
          or None if rendering failed
    """
    from utils.physics_utils import extract_target_point_cloud
    from sampling.analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
    
    # Extract target
    print("[Target] Extracting point cloud...")
    x_tgt, F_tgt = extract_target_point_cloud(target_pc)
    
    # Upsample
    print("[Target] Upsampling to dense surface...")
    mu_tgt, cov_tgt, nrm_tgt, result_tgt = upsample_target(
        x_tgt, F_tgt, rs, 
        export_stages=True,
        output_dir=out_dir  # 🔥 NEW: Pass output directory
    )
    
    print(f"[Target] Upsampled: {len(mu_tgt):,} points")

    print("[Target] Using curvature-based covariance from STAGE 6...")
    
    # Convert cov_tgt to torch for loss computation
    if isinstance(cov_tgt, np.ndarray):
        cov_target_star = torch.from_numpy(cov_tgt).float().cuda()
    else:
        cov_target_star = cov_tgt  # Already torch tensor
    
    # Render
    print("[Target] Rendering...")
    out_tgt = render_target(
        renderer, mu_tgt, cov_tgt, nrm_tgt, campos,
        render_cfg, particle_color,
        cov_target_star=cov_target_star,
        use_curvature_cov=True
    )
    
    # Normalize outputs
    img_tgt, alpha_tgt, depth_tgt, normal_map_tgt = normalize_render_outputs(out_tgt)
    
    # Convert to torch for loss computation
    target_dict = {
        'image': torch.from_numpy(img_tgt).float().cuda() if img_tgt is not None else None,
        'alpha': torch.from_numpy(alpha_tgt).float().cuda() if alpha_tgt is not None else None,
        'depth': torch.from_numpy(depth_tgt).float().cuda() if depth_tgt is not None else None,
        'normal_map': torch.from_numpy(normal_map_tgt).float().cuda() if normal_map_tgt is not None else None,
        'cov_target': cov_target_star,
    }
    
    # Save target renders
    from utils.io_utils import save_target_renders
    save_target_renders(out_dir, img_tgt, alpha_tgt, depth_tgt, normal_map_tgt, result_tgt)
    
    return target_dict


def prepare_rendering_inputs(
    mu: torch.Tensor,
    result: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list
) -> torch.Tensor:
    """
    Prepare RGB colors for rendering using shading.
    
    Args:
        mu: Positions (M, 3)
        result: Upsampling result with normals
        campos: Camera position
        render_cfg: Rendering configuration
        particle_color: Base albedo color
    
    Returns:
        rgb: (M, 3) shaded colors
    """
    # Get normals
    nrm_np = result.get("normals")
    if nrm_np is not None and torch.is_tensor(nrm_np):
        nrm_np = nrm_np.detach().cpu().numpy()
    elif nrm_np is None:
        nrm_np = np.zeros_like(mu.detach().cpu().numpy())
    
    # Compute shading
    rgb_np = compute_shading(
        mu.detach().cpu().numpy(), nrm_np,
        camera_pos=campos,
        light_cfg=render_cfg.get("lighting", {}),
        albedo_color=particle_color,
        model="phong"
    )
    
    rgb = torch.from_numpy(rgb_np).to(mu.device)
    return rgb


# ============================================================================
# Upsampling Wrappers
# ============================================================================

def upsample_current_state(
    pc: Any,
    rs_full: Dict,
    ema_state: Dict,
    seed: int,
    cov_module=None,
    export_stages: bool = False,
    external_levelset=None,
    current_episode: int = 0  # 🔥 NEW: Current episode number
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
    """
    Upsample current simulation state for differentiable rendering.
    
    Args:
        pc: Point cloud from MPM simulation
        rs_full: Upsampling configuration
        ema_state: EMA state (updated in-place)
        seed: Random seed
        cov_module: Optional learnable covariance module
        export_stages: Whether to export stage visualizations (default: False)
        external_levelset: Pre-computed level set
        current_episode: Current episode number (enables morphing mode if >= 0)
    
    Returns:
        Tuple of (mu, cov, result)
          mu: Upsampled positions (M, 3) with gradients
          cov: Covariances (M, 3, 3) with gradients
          result: Full upsampling result dict
    """
    try:
        # 🔥 OPTIMIZED: Use zero-copy views then clone for gradients
        # x = pc.get_positions_torch_view().clone().requires_grad_(True)
        # F = pc.get_def_grads_total_torch_view().clone().requires_grad_(True)
        # Fallback to old method
        try:
            x = pc.get_positions_torch(requires_grad=True)
            F = pc.get_def_grads_total_torch(requires_grad=True)
        except AttributeError:
            print("      ⚠️  PyTorch bindings unavailable")
            return None, None, ema_state
    except AttributeError:
        # Fallback to old method
        try:
            x = pc.get_positions_torch(requires_grad=True)
            F = pc.get_def_grads_total_torch(requires_grad=True)
        except AttributeError:
            print("      ⚠️  PyTorch bindings unavailable")
            return None, None, ema_state
    
    # 🔥 Configure morphing mode
    rs_full_copy = dict(rs_full)
    if 'upsample' not in rs_full_copy:
        rs_full_copy['upsample'] = {}
    if 'covariance' not in rs_full_copy['upsample']:
        rs_full_copy['upsample']['covariance'] = {}
    
    rs_full_copy['upsample']['covariance']['episode'] = current_episode
    
    result = upsample(
        x, F,
        cfg=rs_full_copy,
        state=ema_state,
        seed=seed,
        return_torch=True,
        learnable_cov_module=cov_module,
        export_stages=export_stages,
        external_levelset=external_levelset,
        current_episode=current_episode  # 🔥 Episode별 시각화
    )
    
    mu = result["points"]
    # Use cov_pred for predicted mesh (episode > 0), fallback to cov for backward compat
    cov = result.get("cov_pred", result["cov"])
    
    return mu, cov, result


def compute_render_loss_pass(
    cg: Any,
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
    seed: int,
    cov_module=None,
    external_levelset=None,
    current_episode: int = 0  # 🔥 NEW: Current episode number for morphing mode
) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
    """
    Compute rendering loss and backpropagate gradients.
    
    Pipeline:
      1. Get final state (x, F) from MPM
      2. Upsample: (x, F) → (μ, Σ, normals)
      3. Prepare: RGB from normals + lighting
      4. Render: (μ, Σ, RGB) → {image, alpha, depth}
      5. Loss: Compare with target
      6. Backward: Extract ∂L/∂F, ∂L/∂x
    
    Args:
        cg: Computation graph
        num_timesteps: Number of timesteps
        rs_full: Upsampling configuration
        ema_state: EMA state
        renderer: 3DGS renderer
        loss_manager: Loss manager
        target_render: Ground truth
        view_params: Camera parameters
        campos: Camera position
        render_cfg: Rendering configuration
        particle_color: RGB color
        seed: Random seed
        cov_module: Optional learnable covariance
        external_levelset: Pre-computed level set
        current_episode: Current episode number (enables morphing mode if >= 0)
    
    Returns:
        Tuple of (ema_state, F, x, loss_components)
    """
    import gc
    torch.cuda.empty_cache()
    
    # Get final state
    pc = cg.get_point_cloud(num_timesteps - 1)

    try:
        # 🔥 OPTIMIZED: Use zero-copy views then clone for gradients
        # x = pc.get_positions_torch_view().clone().requires_grad_(True)
        # F = pc.get_def_grads_total_torch_view().clone().requires_grad_(True)
        # Fallback to old method if zero-copy not available
        try:
            x = pc.get_positions_torch(requires_grad=True)
            F = pc.get_def_grads_total_torch(requires_grad=True)
        except AttributeError:
            print("   ⚠️ PyTorch bindings unavailable")
            return None, None, None, None
    except AttributeError:
        # Fallback to old method if zero-copy not available
        try:
            x = pc.get_positions_torch(requires_grad=True)
            F = pc.get_def_grads_total_torch(requires_grad=True)
        except AttributeError:
            print("   ⚠️ PyTorch bindings unavailable")
            return None, None, None, None

    if not x.is_leaf: x.retain_grad()
    if not F.is_leaf: F.retain_grad()
    
    # 🔥 Gradient NaN Shield (워밍업 에피소드용)
    if current_episode <= 3:  # 초기 3 에피소드만
        from utils.covariance_utils import create_nan_clip_hook
        x.register_hook(create_nan_clip_hook("x_mpm", clip_value=5.0))
        F.register_hook(create_nan_clip_hook("F_mpm", clip_value=5.0))
        if current_episode == 0:
            print(f"   🛡️  Gradient NaN shield active (ep {current_episode}, clip=5.0)")
    
    # 🔥 Configure morphing mode
    rs_full_copy = dict(rs_full)
    if 'upsample' not in rs_full_copy:
        rs_full_copy['upsample'] = {}
    if 'covariance' not in rs_full_copy['upsample']:
        rs_full_copy['upsample']['covariance'] = {}
    
    # Set episode for morphing mode (episode >= 0 enables gradient flow)
    rs_full_copy['upsample']['covariance']['episode'] = current_episode
    
    print(f"├─ Morphing mode: episode={current_episode} ({'enabled' if current_episode >= 0 else 'disabled'})")
    
    # Upsample
    with torch.set_grad_enabled(True):
        result = upsample(
            x, F,
            cfg=rs_full_copy,
            state=ema_state,
            seed=seed,
            return_torch=True,
            learnable_cov_module=cov_module,
            external_levelset=external_levelset,
            current_episode=current_episode  # 🔥 Episode별 시각화
        )
    
    mu = result["points"]
    cov = result["cov"]
    ema_state = result["state"]
    
    # 🔥 Store levelset for next pass/episode advection
    if "levelset" in result and result["levelset"] is not None:
        ema_state["levelset"] = result["levelset"]
        print(f"   ✓ Level set stored in ema_state (res={result['levelset'].res}³)")
    
    if mu is None:
        return None, None, None, None
    
    print(f"├─ Upsampled: {len(mu)} points")
    
    # ═══════════════════════════════════════════════════════════════
    # Covariance SPD Enforcement + Voxel Floor (Covariance SPD Enforcement + Floor)
    # ═══════════════════════════════════════════════════════════════
    from utils.covariance_utils import (
        diagnose_covariance_health,
        ensure_spd_with_voxel_floor,
        covariance_regularization_loss
    )
    
    # Diagnose (verbose=False for performance, print only if problem occurs)
    cov_diag_before = diagnose_covariance_health(cov, verbose=False)
    
    # 🔥 Voxel size calculation (from levelset)
    if "levelset" in ema_state and ema_state["levelset"] is not None:
        levelset = ema_state["levelset"]
        bbox_min = levelset.bbox_min
        bbox_max = levelset.bbox_max
        resolution = levelset.res
        
        # Voxel size = scene size / resolution
        scene_size = (bbox_max - bbox_min).max().item()  # maximum dimension
        voxel_size = scene_size / resolution
        
        print(f"│  📐 Voxel size: {voxel_size:.6f} (scene: {scene_size:.2f}, res: {resolution})")
    else:
        # Fallback: approximate default value
        voxel_size = 0.07  # approximate default value
        print(f"│  ⚠️  Levelset not found, voxel_size={voxel_size:.6f} (default)")
    
    # 🔥 Always apply floor (ensure renderable size)
    k_floor = rs_full_copy.get('upsample', {}).get('covariance', {}).get('floor_k', 0.6)
    lambda_min = (k_floor * voxel_size) ** 2
    
    print(f"│  🔧 Covariance Correction (PSD + Floor, EVD-free):")
    print(f"│     λ_min = (k={k_floor:.1f} × voxel)² = {lambda_min:.6f}")
    
    # 🔥 NaN check (before)
    if torch.isnan(cov).any() or torch.isinf(cov).any():
        print(f"│  ❌ NaN/Inf detected in cov BEFORE fix!")
        print(f"│     NaN: {torch.isnan(cov).sum().item()}, Inf: {torch.isinf(cov).sum().item()}")
    
    cov = ensure_spd_with_voxel_floor(
        cov,
        voxel_size=voxel_size,
        k=k_floor
    )
    
    # 🔥 NaN check (after)
    if torch.isnan(cov).any() or torch.isinf(cov).any():
        print(f"│  ❌ NaN/Inf detected in cov AFTER fix!")
        print(f"│     NaN: {torch.isnan(cov).sum().item()}, Inf: {torch.isinf(cov).sum().item()}")
        print(f"│     → EVD Backward Problem Possible!")
    
    # Re-diagnose
    cov_diag_after = diagnose_covariance_health(cov, verbose=False)
    
    print(f"│  ✓ Before: eig ∈ [{cov_diag_before['eig_min']:.2e}, {cov_diag_before['eig_max']:.2e}]")
    print(f"│  ✓ After:  eig ∈ [{cov_diag_after['eig_min']:.2e}, {cov_diag_after['eig_max']:.2e}]")
    print(f"│  ✓ det ∈ [{cov_diag_after['det_min']:.2e}, {cov_diag_after['det_max']:.2e}]")
    
    # 🔥 Verify gradient flow
    if not mu.requires_grad:
        print(f"   ⚠️ WARNING: mu.requires_grad=False (expected True for episode={current_episode})")
        print(f"   → Gradient flow may be broken!")
    else:
        print(f"   ✓ Gradient flow enabled: mu.requires_grad=True")
    
    # Prepare rendering
    with torch.no_grad():
        nrm_np = result.get("normals")
        if nrm_np is not None and torch.is_tensor(nrm_np):
            nrm_np = nrm_np.detach().cpu().numpy()
        elif nrm_np is None:
            nrm_np = np.zeros_like(mu.detach().cpu().numpy())

        rgb_np = compute_shading(
            mu.detach().cpu().numpy(), nrm_np,
            camera_pos=campos,
            light_cfg=render_cfg.get("lighting", {}),
            albedo_color=particle_color,
            model="phong"
        )
        rgb = torch.from_numpy(rgb_np).to(mu.device)
    
    # ========================================================================
    # 🔥 Filtering: φ-mask (월드) + Σ₂D 반경 클램프 (픽셀)
    # ========================================================================
    from sampling.core.levelset_ops import world_to_grid, phi5
    
    # 1) φ-mask: 표면 바깥 샘플 제거 (월드 공간)
    if "levelset" in ema_state and ema_state["levelset"] is not None:
        levelset = ema_state["levelset"]
        sdf5 = phi5(levelset.phi)
        g5 = world_to_grid(mu, levelset.bbox_min, levelset.bbox_max).view(1, -1, 1, 1, 3)
        phi_vals = F_nn.grid_sample(sdf5, g5, mode='bilinear', padding_mode='border', 
                                    align_corners=True).view(-1)
        
        dx = ((levelset.bbox_max - levelset.bbox_min) / (levelset.phi.shape[0] - 1)).max().item()
        tau_phi = 0.12 * dx  # 🔥 0.12·Δx (초반 outlier 과감 제거)
        
        mask_phi = phi_vals.abs() < tau_phi
        
        num_before = mu.shape[0]
        mu = mu[mask_phi]
        cov = cov[mask_phi]
        rgb = rgb[mask_phi]
        
        if hasattr(result, 'get') and result.get('normals') is not None:
            result['normals'] = result['normals'][mask_phi]
        
        num_after = mu.shape[0]
        print(f"│  🔥 φ-mask: {num_before:,} → {num_after:,} (-{num_before-num_after:,}, tau={tau_phi:.4f})")
        
        # 🔥 필터링 후 retain_grad 재호출 (non-leaf → leaf 전환)
        if mu.requires_grad:
            mu.retain_grad()
            cov.retain_grad()
    else:
        mask_phi = None  # φ-mask 미사용
    
    # 2) Σ₂D 반경 클램프: 거대 블러 블롭 제거 (픽셀 공간)
    # 🔥 TODO: Camera transformation 제대로 구현 필요
    # 현재는 비활성화 (간단한 campos 변환은 부정확)
    enable_sigma2d_filter = False  # 🔒 일단 OFF
    
    if enable_sigma2d_filter:
        from utils.covariance_utils import project_sigma3d_to_2d_safe
        
        # TODO: 실제 view matrix로 world → camera 변환 필요
        # mu_cam = (view_matrix @ mu.T).T
        mu_cam = mu - torch.from_numpy(campos).to(mu.device, mu.dtype)
        
        # TODO: 실제 focal length 가져오기
        fx, fy = 400.0, 400.0
        r_px_min = 1.0
        
        S2 = project_sigma3d_to_2d_safe(cov, mu_cam, fx=fx, fy=fy, r_px_min=r_px_min)
        e = torch.linalg.eigvalsh(S2)
        r_px_max = 8.0
        
        mask_size = (e[..., 1] < (r_px_max ** 2))
        
        num_before2 = mu.shape[0]
        mu = mu[mask_size]
        cov = cov[mask_size]
        rgb = rgb[mask_size]
        
        num_after2 = mu.shape[0]
        if num_before2 != num_after2:
            print(f"│  🔥 Σ₂D-clamp: {num_before2:,} → {num_after2:,} (-{num_before2-num_after2:,}, r_max={r_px_max:.1f}px)")
    
    # Render
    pred_render = renderer.render(
        mu, cov, rgb=rgb,
        prefer_cov_precomp=True,
        return_torch=True
    )
    
    # 🔥 cov_target 필터링 (원본은 보존, 복사본 전달)
    cov_target_filtered = None
    if 'cov_target' in target_render and target_render['cov_target'] is not None:
        cov_target_orig = target_render['cov_target']
        if isinstance(cov_target_orig, torch.Tensor):
            if mask_phi is not None:
                # φ-mask로 필터링된 복사본 생성 (원본 보존!)
                cov_target_filtered = cov_target_orig[mask_phi]
            else:
                # 필터링 안 했으면 그대로
                cov_target_filtered = cov_target_orig
    
    # Compute loss (🔥 F 전달: det(F) 바리어 손실 계산)
    render_losses = loss_manager.compute_render_loss(
        pred_render, target_render,
        cov=cov, mu=mu,
        view_params=view_params,
        cov_target=cov_target_filtered,  # 🔥 필터링된 복사본 전달
        F=F  # 🔥 NEW: det(F) 바리어 손실용
    )
    
    loss_render = render_losses['loss_render_total']
    
    # Print losses (🔥 det(F) 바리어 추가)
    print(f"├─ Render loss: {loss_render.item():.6f}")
    for key in ['loss_alpha', 'loss_edge', 'loss_cov_align', 'loss_det_barrier']:
        if key in render_losses:
            val = render_losses[key]
            if torch.is_tensor(val):
                print(f"│  ├─ {key}: {val.item():.6f}")
    
    # 🔥 det(F) 통계 (바리어 효과 확인)
    if F is not None and torch.is_tensor(F):
        with torch.no_grad():
            det_F = torch.det(F)
            det_min = det_F.min().item()
            det_median = det_F.median().item()
            det_max = det_F.max().item()
            det_mean = det_F.mean().item()
            det_below_05 = (det_F < 0.5).float().mean().item()
            det_below_08 = (det_F < 0.8).float().mean().item()
            
            print(f"│  ├─ [det(F) Stats]")
            print(f"│  │  ├─ range: [{det_min:.3f}, {det_max:.3f}], median: {det_median:.3f}, mean: {det_mean:.3f}")
            print(f"│  │  ├─ below 0.5: {det_below_05*100:.1f}% (strong compression)")
            print(f"│  │  └─ below 0.8: {det_below_08*100:.1f}% (moderate compression)")
    
    # Store loss components
    loss_components = {k: v.detach().clone() if torch.is_tensor(v) else v
                      for k, v in render_losses.items()}
    
    # 🔥 NaN check (before backward)
    if torch.isnan(loss_render).any() or torch.isinf(loss_render).any():
        print(f"├─ ❌ NaN/Inf in loss_render BEFORE backward: {loss_render.item()}")
        return None, None, None, None
    
    # Backward
    print(f"├─ Backward pass starting...")
    loss_render.backward()
    
    # 🔥 NaN check (after backward)
    nan_sources = []
    if F.grad is not None and (torch.isnan(F.grad).any() or torch.isinf(F.grad).any()):
        nan_sources.append("F.grad")
        print(f"│  ❌ NaN/Inf in F.grad: NaN={torch.isnan(F.grad).sum().item()}, Inf={torch.isinf(F.grad).sum().item()}")
    
    if x.grad is not None and (torch.isnan(x.grad).any() or torch.isinf(x.grad).any()):
        nan_sources.append("x.grad")
        print(f"│  ❌ NaN/Inf in x.grad: NaN={torch.isnan(x.grad).sum().item()}, Inf={torch.isinf(x.grad).sum().item()}")
    
    # 🔥 mu, cov는 필터링 후 non-leaf일 수 있어서 .grad 체크 스킵
    # (중요한 건 x.grad, F.grad이므로 문제없음)
    
    if nan_sources:
        print(f"└─ ❌ Backward produced NaN in: {', '.join(nan_sources)}")
        print(f"   → Possible: EVD Backward, normalize(0), z-division")
        return None, None, None, None
    else:
        print(f"└─ ✅ Backward completed, no NaN detected")
    
    return ema_state, F, x, loss_components


def _visibility_mask(
    mu: Optional[torch.Tensor],
    view_params: Optional[Dict],
    alpha_tgt: Optional[torch.Tensor],
    edge_info: Optional[Dict] = None,
    enabled: bool = False
) -> Optional[torch.Tensor]:
    """
    🔥 가시성·경계 마스킹 (선택적).
    
    비가시/저경계 픽셀의 렌더 그라디언트를 희석/마스킹하여
    잡음성 그라디언트 주입을 억제합니다.
    
    전략:
    - 스크린 안/앞쪽(z>0) 체크
    - (선택) 경계 강도 기반 가중
    
    Args:
        mu: (N, 3) 3D positions
        view_params: Camera parameters
        alpha_tgt: Target alpha map
        edge_info: Edge alignment info (선택)
        enabled: 마스킹 활성화 여부 (기본: False)
    
    Returns:
        (N,) 가중치 또는 None (비활성화 시)
    """
    if not enabled or mu is None:
        return None
    
    N = mu.shape[0]
    device = mu.device
    
    # 간단 스텁: 모두 1 반환 (향후 확장 가능)
    # TODO: 실루엣/경계 강도 기반 가중 구현
    # - edge_info['edge_grad_norm_mean'] 등 활용
    # - alpha_tgt의 그라디언트 맵 활용
    
    return None  # 현재는 비활성화 (향후 점진적 적용 권장)


def extract_render_gradients(
    F: torch.Tensor, 
    x: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    view_params: Optional[Dict] = None,
    target_render: Optional[Dict] = None,
    visibility_masking: bool = False
) -> Optional[Dict]:
    """
    Extract render gradients for injection to physics.
    
    🔥 개선: 가시성·경계 마스킹 지원 (선택적)
    
    Args:
        F: Deformation gradients with .grad
        x: Positions with .grad
        mu: (N, 3) 3D positions (마스킹용, 선택)
        view_params: Camera parameters (마스킹용, 선택)
        target_render: Target render dict (마스킹용, 선택)
        visibility_masking: 마스킹 활성화 여부 (기본: False)
    
    Returns:
        Dict with {dLdF, dLdx} or None if extraction failed
    """
    if F.grad is None:
        print(f"└─ ⚠️ F.grad is None, cannot extract gradients")
        return None
    
    dLdF = F.grad.detach().cpu().numpy().astype(np.float32)
    dLdx = x.grad.detach().cpu().numpy().astype(np.float32) if x.grad is not None else None
    
    if dLdx is None:
        dLdx = np.zeros_like(dLdF[:, :, 0])
    
    # 🔥 가시성·경계 마스킹 (선택적)
    if visibility_masking and mu is not None:
        alpha_tgt = target_render.get('alpha') if target_render is not None else None
        mask = _visibility_mask(mu, view_params, alpha_tgt, enabled=True)
        
        if mask is not None:
            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            # dLdF: (N,3,3), dLdx: (N,3), mask: (N,)
            dLdF *= mask_np[:, None, None]
            dLdx *= mask_np[:, None]
            
            print(f"   🔎 Visibility masking applied (mean weight: {mask_np.mean():.3f})")
    
    return {
        'dLdF': dLdF,
        'dLdx': dLdx
    }


__all__ = [
    'setup_renderer',
    'upsample_target',
    'render_target',
    'normalize_render_outputs',
    'create_target_render',
    'prepare_rendering_inputs',
    'upsample_current_state',
    'compute_render_loss_pass',
    'extract_render_gradients',
]

