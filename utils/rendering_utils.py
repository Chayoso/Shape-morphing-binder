"""
Rendering Utilities - 3D Gaussian Splatting & Target Rendering

Handles renderer setup, target rendering, and loss computation.
"""

import numpy as np
import torch
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
    render_cfg: Dict
) -> Tuple[Optional[Any], Dict]:
    """
    Initialize 3D Gaussian Splatting renderer with camera configuration.
    
    Args:
        cam_cfg: Camera configuration (resolution, focal length, pose)
        render_cfg: Rendering configuration (bg color, scale_modifier)
    
    Returns:
        Tuple of (renderer, view_params)
          renderer: GSRenderer3DGS instance or None if failed
          view_params: Camera view matrices and parameters
    """
    try:
        W, H, tanfovx, tanfovy, view_T, proj_T, campos = make_matrices_from_yaml(cam_cfg)
        
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
    export_stages: bool = True
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
    
    Returns:
        Tuple of (mu_tgt, cov_tgt, nrm_tgt, result_tgt)
    """
    # 🔥 Set episode=-1 for target mesh (initial frame)
    rs["episode"] = -1
    
    result_tgt = upsample(
        x_tgt, F_tgt,
        cfg=rs,               
        seed=9999,
        return_torch=False,
        export_stages=export_stages
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
    mu_tgt, cov_tgt, nrm_tgt, result_tgt = upsample_target(x_tgt, F_tgt, rs, export_stages=True)
    
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
    cov_module=None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
    """
    Upsample current simulation state for differentiable rendering.
    
    Args:
        pc: Point cloud from MPM simulation
        rs_full: Upsampling configuration
        ema_state: EMA state (updated in-place)
        seed: Random seed
        cov_module: Optional learnable covariance module
    
    Returns:
        Tuple of (mu, cov, result)
          mu: Upsampled positions (M, 3) with gradients
          cov: Covariances (M, 3, 3) with gradients
          result: Full upsampling result dict
    """
    try:
        x = pc.get_positions_torch(requires_grad=True)
        F = pc.get_def_grads_total_torch(requires_grad=True)
    except AttributeError:
        print("      ⚠️  PyTorch bindings unavailable")
        return None, None, ema_state
    
    result = upsample(
        x, F,
        cfg=rs_full,
        state=ema_state,
        seed=seed,
        return_torch=True,
        learnable_cov_module=cov_module
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
    cov_module=None
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
    
    Returns:
        Tuple of (ema_state, F, x, loss_components)
    """
    import gc
    torch.cuda.empty_cache()
    
    # Get final state
    pc = cg.get_point_cloud(num_timesteps - 1)
    
    try:
        x = pc.get_positions_torch(requires_grad=True)
        F = pc.get_def_grads_total_torch(requires_grad=True)
    except AttributeError:
        print("   ⚠️ PyTorch bindings unavailable")
        return None, None, None, None
    
    if not x.is_leaf: x.retain_grad()
    if not F.is_leaf: F.retain_grad()
    
    # Upsample
    with torch.set_grad_enabled(True):
        result = upsample(
            x, F,
            cfg=rs_full,
            state=ema_state,
            seed=seed,
            return_torch=True,
            learnable_cov_module=cov_module
        )
    
    mu = result["points"]
    cov = result["cov"]
    ema_state = result["state"]
    
    if mu is None:
        return None, None, None, None
    
    print(f"├─ Upsampled: {len(mu)} points")
    
    mu.retain_grad()
    cov.retain_grad()
    
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
    
    # Render
    pred_render = renderer.render(
        mu, cov, rgb=rgb,
        prefer_cov_precomp=True,
        return_torch=True
    )
    
    # Compute loss
    render_losses = loss_manager.compute_render_loss(
        pred_render, target_render,
        cov=cov, mu=mu,
        view_params=view_params,
        cov_target=target_render.get('cov_target')
    )
    
    loss_render = render_losses['loss_render_total']
    
    # Print losses
    print(f"├─ Render loss: {loss_render.item():.6f}")
    for key in ['loss_alpha', 'loss_edge', 'loss_cov_align']:
        if key in render_losses:
            val = render_losses[key]
            if torch.is_tensor(val):
                print(f"│  ├─ {key}: {val.item():.6f}")
    
    # Store loss components
    loss_components = {k: v.detach().clone() if torch.is_tensor(v) else v
                      for k, v in render_losses.items()}
    
    # Backward
    loss_render.backward()
    
    return ema_state, F, x, loss_components


def extract_render_gradients(F: torch.Tensor, x: torch.Tensor) -> Optional[Dict]:
    """
    Extract render gradients for injection to physics.
    
    Args:
        F: Deformation gradients with .grad
        x: Positions with .grad
    
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

