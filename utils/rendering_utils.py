"""
Rendering Utilities - 3D Gaussian Splatting & Target Rendering

Handles renderer setup, target rendering, and loss computation.
"""

import numpy as np
import copy
import torch
import torch.nn.functional as F_nn
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from renderer import GSRenderer3DGS, make_matrices_from_yaml, compute_shading
from sampling import upsample


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BG_COLOR = [1.0, 1.0, 1.0]
DEFAULT_PARTICLE_COLOR = [0.27, 0.51, 0.71]


def _save_gradient_focus_debug(mu: torch.Tensor, mask: torch.Tensor,
                               view_params: Dict, campos: np.ndarray,
                               output_path: Path) -> None:
    """Save 3D scatter showing where gradients are active."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mu_np = mu.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        grad_pts = mu_np[mask_np]
        vol_pts = mu_np[~mask_np]
        campos_np = np.asarray(campos, dtype=np.float32)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        if grad_pts.size > 0:
            ax.scatter(grad_pts[:, 0], grad_pts[:, 1], grad_pts[:, 2],
                       s=1, c="#ff4d4f", alpha=0.85)
        view_T = view_params.get('view_T')
        tanfovx = view_params.get('tanfovx', 1.0)
        tanfovy = view_params.get('tanfovy', 1.0)
        if view_T is not None:
            view_T = np.asarray(view_T, dtype=np.float32)
            w2c = view_T.T
            c2w = np.linalg.inv(w2c)
            right = c2w[:3, 0]
            up = c2w[:3, 1]
            forward = -c2w[:3, 2]
            right /= np.linalg.norm(right) + 1e-8
            up /= np.linalg.norm(up) + 1e-8
            forward /= np.linalg.norm(forward) + 1e-8
        else:
            target = view_params.get('target', campos_np + np.array([0, 0, -1], dtype=np.float32))
            forward = target - campos_np
            forward /= np.linalg.norm(forward) + 1e-8
            up = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right) + 1e-8
            up = np.cross(right, forward)
            up /= np.linalg.norm(up) + 1e-8

        depth = float(view_params.get('debug_depth', 8.0))
        center = campos_np + forward * depth
        corner_dirs = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                dir_vec = forward + sx * tanfovx * right + sy * tanfovy * up
                dir_vec /= np.linalg.norm(dir_vec) + 1e-8
                corner_dirs.append(campos_np + depth * dir_vec)

        frustum_edges = []
        for corner in corner_dirs:
            frustum_edges.append(np.stack([campos_np, corner], axis=0))
        corner_idx = [0, 1, 3, 2]
        for i in range(4):
            frustum_edges.append(np.stack([corner_dirs[corner_idx[i]],
                                           corner_dirs[corner_idx[(i+1)%4]]], axis=0))

        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        lc = Line3DCollection(frustum_edges, colors='black', linewidths=0.8, alpha=0.6)
        ax.add_collection3d(lc)
        elev = np.degrees(np.arctan2(forward[2], np.linalg.norm(forward[:2]) + 1e-8))
        azim = np.degrees(np.arctan2(forward[1], forward[0]))

        ax.scatter(*campos_np, c='black', marker='^', s=60)
        ax.set_title("Visible Surface (Grad Active)")
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)

        ax2 = fig.add_subplot(122, projection='3d')
        if vol_pts.size > 0:
            ax2.scatter(vol_pts[:, 0], vol_pts[:, 1], vol_pts[:, 2],
                        s=1, c="#1f9cff", alpha=0.35)
        lc2 = Line3DCollection(frustum_edges, colors='black', linewidths=0.8, alpha=0.6)
        ax2.add_collection3d(lc2)
        ax2.scatter(*campos_np, c='black', marker='^', s=60)
        ax2.set_title("Volume (Grad=0)")
        ax2.set_axis_off()
        ax2.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"│  📸 Saved gradient debug → {output_path}")
    except Exception as e:
        print(f"│  ⚠️  Gradient debug visualization failed: {e}")


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
    rs_local = copy.deepcopy(rs)
    if "covariance" not in rs_local:
        rs_local["covariance"] = {}
    rs_local["covariance"]["episode"] = -1
    
    # 🔥 Set output_dir for stage exports
    if output_dir is not None and "debug" not in rs_local:
        rs_local["debug"] = {}
    if output_dir is not None:
        rs_local["debug"]["output_dir"] = str(output_dir)
    
    result_tgt = upsample(
        x_tgt, F_tgt,
        cfg=rs_local,               
        seed=9999,
        return_torch=False,
        export_stages=export_stages,
        current_episode=-1  # 🔥 NEW: Mark as target
    )
    
    mu_tgt = result_tgt["points"]
    cov_tgt = result_tgt["cov"]
    cov_target_star = result_tgt.get("cov_target")
    nrm_tgt = result_tgt.get("normals")
    
    if cov_target_star is None:
        cov_target_star = cov_tgt
    
    result_tgt["cov_target"] = cov_target_star
    
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

    # 🔥 Orient normals toward camera (for correct normal map visualization)
    # All visible surfaces should appear pink/magenta in the normal map
    view_dir = campos - mu_tgt  # (N, 3) vector from particle to camera
    view_dir_norm = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8)

    # Flip normals that point away from camera
    dot_product = np.sum(nrm_tgt * view_dir_norm, axis=1)  # (N,)
    flip_mask = dot_product < 0
    nrm_tgt = nrm_tgt.copy()  # Don't modify original
    nrm_tgt[flip_mask] = -nrm_tgt[flip_mask]

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
    
    # Extract target
    print("[Target] Extracting point cloud...", flush=True)
    x_tgt, F_tgt = extract_target_point_cloud(target_pc)

    # Upsample
    print("[Target] Upsampling to dense surface...", flush=True)
    mu_tgt, cov_tgt, nrm_tgt, result_tgt = upsample_target(
        x_tgt, F_tgt, rs,
        export_stages=True,
        output_dir=out_dir  # 🔥 NEW: Pass output directory
    )

    print(f"[Target] Upsampled: {len(mu_tgt):,} points", flush=True)

    print("[Target] Using curvature-based covariance from STAGE 6...", flush=True)

    # Convert cov_tgt to torch for loss computation
    cov_target_star = result_tgt.get("cov_target", cov_tgt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(cov_target_star, np.ndarray):
        cov_target_star = torch.from_numpy(cov_target_star).float().to(device)
    elif isinstance(cov_target_star, torch.Tensor):
        cov_target_star = cov_target_star.to(device)

    # Render
    print("[Target] Rendering...", flush=True)
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
    
    physics_grid = rs_full_copy.get("physics_grid", {})
    voxel_size = float(physics_grid.get("grid_dx", 0.07))
    grid_min = physics_grid.get("grid_min")
    grid_max = physics_grid.get("grid_max")
    if grid_min is not None and grid_max is not None and len(grid_min) == 3 and len(grid_max) == 3:
        grid_min_t = torch.tensor(grid_min, dtype=torch.float32)
        grid_max_t = torch.tensor(grid_max, dtype=torch.float32)
        scene_size = (grid_max_t - grid_min_t).max().item()
        print(f"│  📐 Voxel size: {voxel_size:.6f} (scene: {scene_size:.2f}, grid_dx={voxel_size:.2f})")
    else:
        print(f"│  📐 Voxel size: {voxel_size:.6f} (grid_dx)")
    
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
    
    # Surface classification with multiple strategies
    mask_settings = render_cfg.get("surface_masking", {})
    method = str(mask_settings.get("method", "det")).lower()
    surface_mask = None
    num_total = mu.shape[0]

    def _as_tensor(data):
        if data is None:
            return None
        if torch.is_tensor(data):
            return data.to(mu.device)
        try:
            return torch.from_numpy(data).to(mu.device)
        except Exception:
            return None

    def _det_surface_mask():
        F_field = result.get("F_interp")
        F_field_t = _as_tensor(F_field)
        if F_field_t is None or F_field_t.shape[0] != num_total:
            F_field_t = _as_tensor(F)
            if F_field_t is not None and F_field_t.shape[0] != num_total:
                F_field_t = None
        if F_field_t is None:
            print("│  ⚠️  det(F) mask skipped (F_interp unavailable)")
            return None

        det_center = float(mask_settings.get("det_center", 1.0))
        det_tau = float(mask_settings.get("det_tau", 0.08))
        det_vals = torch.det(F_field_t.detach())
        det_offset = torch.abs(det_vals - det_center)

        mask = det_offset <= det_tau
        min_ratio = float(mask_settings.get("det_min_ratio", 0.05))
        min_points = int(mask_settings.get("det_min_points", 1000))
        keep_min = max(int(min_ratio * num_total), min_points)
        keep_min = min(max(keep_min, 1), num_total) if num_total > 0 else 0

        num_surface = mask.sum().item()
        if num_total > 0 and num_surface < keep_min:
            kth = min(keep_min, num_total)
            kth_val = torch.kthvalue(det_offset, kth).values.item()
            det_tau = max(det_tau, kth_val + 1e-6)
            mask = det_offset <= det_tau
            num_surface = mask.sum().item()
            print(f"│  ⚠️  det(F) τ relaxed → keep {num_surface:,}/{num_total:,} (τ={det_tau:.4f})")

        if num_total > 0:
            print("│  🔍 det(F) stats:")
            print(f"│     range: [{det_vals.min().item():.3f}, {det_vals.max().item():.3f}] "
                  f"mean: {det_vals.mean().item():.3f}")
            print(f"│     center={det_center:.3f}, τ={det_tau:.4f}, ratio={(mask.float().mean().item()*100):.2f}%")
        return mask

    def _edge_surface_mask():
        alpha_target = target_render.get('alpha')
        if alpha_target is None or not view_params:
            print("│  ⚠️  Edge mask skipped (alpha/view params unavailable)")
            return None

        if isinstance(alpha_target, np.ndarray):
            alpha_tensor = torch.from_numpy(alpha_target)
        elif torch.is_tensor(alpha_target):
            alpha_tensor = alpha_target.clone()
        else:
            alpha_tensor = None
        if alpha_tensor is None:
            print("│  ⚠️  Edge mask skipped (alpha tensor invalid)")
            return None

        alpha_tensor = alpha_tensor.to(mu.device, dtype=torch.float32)
        if alpha_tensor.ndim == 3:
            if alpha_tensor.shape[0] == 1:
                alpha_tensor = alpha_tensor[0]
            elif alpha_tensor.shape[-1] == 1:
                alpha_tensor = alpha_tensor[..., 0]
        alpha_tensor = alpha_tensor.unsqueeze(0).unsqueeze(0)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=mu.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=mu.device
        ).view(1, 1, 3, 3)

        grad_x = F_nn.conv2d(alpha_tensor, sobel_x, padding=1)
        grad_y = F_nn.conv2d(alpha_tensor, sobel_y, padding=1)
        grad_norm = torch.sqrt(torch.clamp(grad_x**2 + grad_y**2, min=1e-12))

        view_T = view_params.get('view_T')
        tanfovx = view_params.get('tanfovx')
        tanfovy = view_params.get('tanfovy')
        if view_T is None or tanfovx is None or tanfovy is None:
            print("│  ⚠️  Edge mask skipped (missing view parameters)")
            return None
        if isinstance(view_T, np.ndarray):
            view_T = torch.from_numpy(view_T)
        view_T = view_T.to(mu.device, dtype=mu.dtype)

        ones = torch.ones(num_total, 1, device=mu.device, dtype=mu.dtype)
        mu_hom = torch.cat([mu, ones], dim=1)
        mu_cam = (view_T @ mu_hom.T).T

        z = torch.clamp(mu_cam[:, 2], min=1e-4)
        ndc_x = mu_cam[:, 0] / z
        ndc_y = mu_cam[:, 1] / z

        screen_x = (ndc_x / tanfovx * 0.5 + 0.5).clamp(0.0, 1.0)
        screen_y = (-ndc_y / tanfovy * 0.5 + 0.5).clamp(0.0, 1.0)

        grid_x = screen_x * 2 - 1
        grid_y = screen_y * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, num_total, 1, 2)

        grad_sampled = F_nn.grid_sample(
            grad_norm, grid, align_corners=False, mode='bilinear'
        ).view(-1)

        percentile = float(mask_settings.get("edge_percentile", 0.9))
        percentile = min(max(percentile, 0.0), 0.999)
        grad_thresh = torch.quantile(grad_sampled, percentile)
        edge_threshold = float(mask_settings.get("edge_threshold", grad_thresh.item()))

        mask = grad_sampled > edge_threshold
        min_ratio = float(mask_settings.get("edge_min_ratio", 0.1))
        min_points = int(mask_settings.get("edge_min_points", 2000))
        keep_min = max(int(min_ratio * num_total), min_points)
        keep_min = min(max(keep_min, 1), num_total) if num_total > 0 else 0

        num_surface = mask.sum().item()
        if num_total > 0 and num_surface < keep_min:
            kth = max(1, num_total - keep_min + 1)
            kth_val = torch.kthvalue(grad_sampled, kth).values.item()
            edge_threshold = min(edge_threshold, kth_val - 1e-8)
            mask = grad_sampled >= edge_threshold
            num_surface = mask.sum().item()
            print(f"│  ⚠️  Edge threshold relaxed → keep {num_surface:,}/{num_total:,} (τ={edge_threshold:.4e})")

        if num_total > 0:
            print("│  🔍 Edge mask stats:")
            print(f"│     grad range: [{grad_sampled.min().item():.2e}, {grad_sampled.max().item():.2e}]")
            print(f"│     threshold: {edge_threshold:.2e}, ratio={(mask.float().mean().item()*100):.2f}%")

        debug_png = mask_settings.get("edge_debug_png")
        if debug_png:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                grad_np = grad_norm.squeeze().detach().cpu().numpy()
                plt.figure(figsize=(6, 5))
                plt.imshow(grad_np, cmap='hot')
                plt.title("Alpha gradient magnitude")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.savefig(debug_png, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"│  🔎 Saved edge gradient heatmap → {debug_png}")
            except Exception as e:
                print(f"│  ⚠️  Edge heatmap save failed: {e}")
        return mask

    if method == "det":
        surface_mask = _det_surface_mask()
    elif method == "edge":
        surface_mask = _edge_surface_mask()
    else:
        print(f"│  ⚠️  Unknown surface mask method '{method}', using fallback ratio")
    
    # Render
    pred_render = renderer.render(
        mu, cov, rgb=rgb,
        prefer_cov_precomp=True,
        return_torch=True
    )
    
    # 🔥 cov_target 필터링 (원본은 보존, 복사본 전달)
    def _apply_mask_to_tensor(tensor, mask, label):
        if tensor is None or mask is None:
            return tensor
        if isinstance(mask, torch.Tensor):
            mask_len = mask.shape[0]
        else:
            mask_len = len(mask)
        tensor_len = tensor.shape[0]
        if tensor_len != mask_len:
            print(f"│  ⚠️  Mask length mismatch for {label}: tensor={tensor_len}, mask={mask_len}")
            return tensor
        if isinstance(tensor, torch.Tensor):
            return tensor[mask].clone()
        else:
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
            return tensor[mask_np].copy()

    def _depth_surface_mask():
        depth_thresh = mask_settings.get("depth_threshold")
        if depth_thresh is None:
            return None
        depth_pred = pred_render.get('depth')
        depth_tgt = target_render.get('depth')
        if depth_pred is None or depth_tgt is None or not view_params:
            print("│  ⚠️  Depth mask skipped (missing depth/view params)")
            return None

        def _to_tensor_depth(arr):
            if torch.is_tensor(arr):
                return arr.to(mu.device, dtype=torch.float32)
            try:
                return torch.from_numpy(arr).to(mu.device, dtype=torch.float32)
            except Exception:
                return None

        depth_pred_t = _to_tensor_depth(depth_pred)
        depth_tgt_t = _to_tensor_depth(depth_tgt)
        if depth_pred_t is None or depth_tgt_t is None:
            print("│  ⚠️  Depth mask skipped (invalid depth tensors)")
            return None

        if depth_pred_t.ndim == 3:
            depth_pred_t = depth_pred_t[0]
        if depth_tgt_t.ndim == 3:
            depth_tgt_t = depth_tgt_t[0]
        depth_pred_t = depth_pred_t.unsqueeze(0).unsqueeze(0)
        depth_tgt_t = depth_tgt_t.unsqueeze(0).unsqueeze(0)

        view_T = view_params.get('view_T')
        tanfovx = view_params.get('tanfovx')
        tanfovy = view_params.get('tanfovy')
        if view_T is None or tanfovx is None or tanfovy is None:
            print("│  ⚠️  Depth mask skipped (missing camera info)")
            return None
        if isinstance(view_T, np.ndarray):
            view_T = torch.from_numpy(view_T)
        view_T = view_T.to(mu.device, dtype=mu.dtype)

        ones = torch.ones(num_total, 1, device=mu.device, dtype=mu.dtype)
        mu_hom = torch.cat([mu, ones], dim=1)
        mu_cam = (view_T @ mu_hom.T).T
        z = torch.clamp(mu_cam[:, 2], min=1e-4)
        ndc_x = mu_cam[:, 0] / z
        ndc_y = mu_cam[:, 1] / z
        grid_x = (ndc_x / tanfovx * 0.5 + 0.5).clamp(0.0, 1.0) * 2 - 1
        grid_y = (-ndc_y / tanfovy * 0.5 + 0.5).clamp(0.0, 1.0) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, num_total, 1, 2)

        depth_pred_s = F_nn.grid_sample(depth_pred_t, grid, align_corners=False).view(-1)
        depth_tgt_s = F_nn.grid_sample(depth_tgt_t, grid, align_corners=False).view(-1)
        depth_diff = torch.abs(depth_pred_s - depth_tgt_s)

        depth_thresh = float(depth_thresh)
        depth_mask = depth_diff <= depth_thresh
        min_ratio = float(mask_settings.get("depth_min_ratio", 0.1))
        keep_min = max(int(min_ratio * num_total), 1)
        keep_min = min(keep_min, num_total)
        num_depth = depth_mask.sum().item()
        if num_total > 0 and num_depth < keep_min:
            kth = min(keep_min, num_total)
            kth_val = torch.kthvalue(depth_diff, kth).values.item()
            depth_thresh = max(depth_thresh, kth_val + 1e-6)
            depth_mask = depth_diff <= depth_thresh
            num_depth = depth_mask.sum().item()
            print(f"│  ⚠️  Depth threshold relaxed → keep {num_depth:,}/{num_total:,} (τ={depth_thresh:.4f})")

        if num_total > 0:
            print("│  🔍 Depth mask stats:")
            print(f"│     diff range: [{depth_diff.min().item():.3e}, {depth_diff.max().item():.3e}]")
            print(f"│     threshold: {depth_thresh:.3f}, ratio={(depth_mask.float().mean().item()*100):.2f}%")
        return depth_mask

    depth_mask = _depth_surface_mask()
    if depth_mask is not None:
        if surface_mask is None:
            surface_mask = depth_mask
        else:
            surface_mask = surface_mask & depth_mask
        print(f"│  🔗 Edge ∧ depth ratio: {(surface_mask.float().mean().item()*100):.2f}%")

    cov_target_filtered = None
    cov_target_orig = target_render.get('cov_target')
    if cov_target_orig is not None:
        if isinstance(cov_target_orig, torch.Tensor):
            cov_target_filtered = cov_target_orig.clone()
        else:
            cov_target_filtered = np.copy(cov_target_orig)

    # 🔥 CRITICAL FIX: Guarantee a surface mask exists
    if surface_mask is None:
        num_total = mu.shape[0]
        target_surface_ratio = render_cfg.get('surface_mask_ratio', 0.15)
        surface_mask_mode = render_cfg.get('surface_mask_mode', 'last')
        num_surface = max(1, int(num_total * target_surface_ratio))

        surface_mask = torch.zeros(num_total, dtype=torch.bool, device=mu.device)
        if surface_mask_mode == 'last':
            surface_mask[-num_surface:] = True
        else:
            surface_mask[:num_surface] = True

        actual_ratio = num_surface / max(1, num_total)
        if surface_mask_mode == 'last':
            start_idx = max(0, num_total - num_surface)
            print(f"   Strategy:           LAST {actual_ratio*100:.1f}% (idx {start_idx:,} → {num_total:,})")
        else:
            print(f"   Strategy:           FIRST {actual_ratio*100:.1f}% (idx 0 → {num_surface:,})")
        print("="*60)
        print("🔥 SURFACE MASK (RATIO FALLBACK)")
        print(f"   Total particles:    {num_total:,}")
        print(f"   Surface particles:  {num_surface:,} ({actual_ratio*100:.1f}%)")
        print(f"   Volume particles:   {num_total - num_surface:,} ({(1-actual_ratio)*100:.1f}%)")
        print(f"   Config ratio:       {target_surface_ratio} (render.surface_mask_ratio)")
        print(f"   Mode:               '{surface_mask_mode}' (render.surface_mask_mode)")
        print("="*60)
    else:
        ratio_current = surface_mask.float().mean().item()
        print(f"│  🔎 Surface mask ready: {surface_mask.sum().item():,}/{surface_mask.shape[0]:,} "
              f"({ratio_current*100:.2f}%)")
        warn_min = float(render_cfg.get("surface_mask_warn_min", 0.01))
        warn_max = float(render_cfg.get("surface_mask_warn_max", 0.8))
        if ratio_current < warn_min:
            print(f"│  ⚠️  Surface mask ratio below {warn_min*100:.1f}% → gradients may vanish")
        if ratio_current > warn_max:
            print(f"│  ⚠️  Surface mask ratio above {warn_max*100:.1f}% → gradients may dominate volume")

    visible_mask = None
    visible_ratio_value = None
    grad_focus_cfg = render_cfg.get("gradient_focus", {})
    visible_only = bool(grad_focus_cfg.get("visible_only", False))
    if visible_only and surface_mask is not None:
        min_ratio = float(grad_focus_cfg.get("min_ratio", 0.05))
        min_points = int(grad_focus_cfg.get("min_points", 5000))
        vis_count = surface_mask.sum().item()
        required = max(int(min_ratio * surface_mask.shape[0]), min_points)
        required = min(max(required, 1), surface_mask.shape[0])
        view_params_focus = dict(view_params)
        if vis_count < required:
            print(f"│  ⚠️  Visible mask smaller than requested ({vis_count:,}/{required:,})")
        visible_mask = surface_mask.clone()
        visible_ratio = visible_mask.float().mean().item()
        warn_vis_min = float(grad_focus_cfg.get("warn_ratio_min", 0.02))
        warn_vis_max = float(grad_focus_cfg.get("warn_ratio_max", 0.7))
        fallback_triggered = False
        if visible_ratio < warn_vis_min:
            print(f"│  ⚠️  Visible ratio below {warn_vis_min*100:.1f}% → disabling gradient focus for this pass")
            visible_only = False
            visible_mask = None
            fallback_triggered = True
        elif visible_ratio > warn_vis_max:
            target = max(int(warn_vis_max * surface_mask.shape[0]), min_points)
            target = min(max(target, 1), surface_mask.shape[0])
            if target < vis_count:
                idx = torch.nonzero(visible_mask, as_tuple=False).view(-1)
                if idx.numel() > 0:
                    perm = torch.randperm(idx.numel(), device=visible_mask.device)
                    keep = idx[perm[:target]]
                    new_mask = torch.zeros_like(visible_mask)
                    new_mask[keep] = True
                    visible_mask = new_mask
                    visible_ratio = visible_mask.float().mean().item()
                    print(f"│  ⚠️  Visible ratio above {warn_vis_max*100:.1f}% → subsampling to {visible_ratio*100:.2f}%")
                    fallback_triggered = True

        if visible_only and visible_mask is not None:
            visible_ratio_value = visible_mask.float().mean().item()
            print(f"│  🎯 Gradient focus enabled: restrict grads to {int(visible_mask.sum().item()):,} particles ({visible_ratio_value*100:.2f}%)")
        elif fallback_triggered:
            print("│  🎯 Gradient focus fallback engaged (using full set)")
            visible_mask = None
        debug_png = grad_focus_cfg.get("debug_png")
        if debug_png and visible_only and visible_mask is not None:
            debug_path = Path(str(debug_png).format(ep=current_episode))
            _save_gradient_focus_debug(mu, visible_mask, view_params_focus, campos, debug_path)


    # ───────────────────────────────────────────────────────────────
    # Surface debug visualization (after mask finalized)
    # ───────────────────────────────────────────────────────────────
    surface_debug = render_cfg.get("surface_debug", {})
    if surface_debug.get("enabled", False):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            mu_np = mu.detach().cpu().numpy()
            surface_mask_np = surface_mask.detach().cpu().numpy().astype(bool)

            slice_cfg = surface_debug.get("slice", {})
            slice_axis = slice_cfg.get("axis", "z")
            axis_map = {"x": 0, "y": 1, "z": 2}
            axis_idx = axis_map.get(str(slice_axis).lower(), 2)
            center = slice_cfg.get("center")
            if center is None:
                center = float(np.median(mu_np[:, axis_idx]))
            thickness = float(slice_cfg.get("thickness", 1.0))
            slice_mask = np.abs(mu_np[:, axis_idx] - center) <= thickness
            if not slice_mask.any():
                slice_mask[:] = True  # fallback to all points

            mu_slice = mu_np[slice_mask]
            surf_slice = surface_mask_np[slice_mask]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].set_title(f"Surface points ({slice_axis}-slice)")
            axes[1].set_title(f"Volume points ({slice_axis}-slice)")
            if surf_slice.any():
                axes[0].scatter(mu_slice[surf_slice, 0], mu_slice[surf_slice, 1],
                                c='red', s=3, alpha=0.6)
            if (~surf_slice).any():
                axes[1].scatter(mu_slice[~surf_slice, 0], mu_slice[~surf_slice, 1],
                                c='blue', s=3, alpha=0.4)
            for ax in axes:
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_aspect("equal")
            plt.tight_layout()

            output_png = surface_debug.get("output_png")
            if output_png:
                out_path = Path(output_png)
            else:
                out_path = Path(surface_debug.get("output_dir", "output")) / f"surface_debug_ep{current_episode:03d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Surface debug visualization failed: {e}")

    if surface_mask is None:
        surface_mask = torch.ones(mu.shape[0], dtype=torch.bool, device=mu.device)

    # 🔥 CRITICAL: Filter cov_target to match surface mask
    # Reasoning:
    # - Target was upsampled to 150k particles from surface
    # - cov_target (Σ★) computed from curvature for all 150k
    # - BUT: surface_mask identifies which are TRUE surface (e.g., 15k)
    # - So: cov_target should only exist for those 15k surface particles
    # - Volume/interior particles shouldn't have target covariances
    if cov_target_filtered is not None and surface_mask is not None:
        cov_target_filtered = _apply_mask_to_tensor(
            cov_target_filtered, surface_mask, "surface mask cov_target"
        )
        if cov_target_filtered is not None:
            print(f"│  🔥 cov_target filtered to surface: {cov_target_filtered.shape[0]:,} entries")

    if cov_target_filtered is not None and not isinstance(cov_target_filtered, torch.Tensor):
        cov_target_filtered = torch.from_numpy(cov_target_filtered).to(mu.device)

    # 🔥 NEW: Create opacity tensor for shrinkage regularization
    # Start with all particles fully opaque (α=1.0)
    # Shrinkage loss will gradually reduce interior particle opacity
    opacity = torch.ones(mu.shape[0], dtype=torch.float32, device=mu.device, requires_grad=True)

    # Compute loss (🔥 F 전달: det(F) 바리어 손실 계산)
    loss_visibility_ratio = None
    if grad_focus_cfg.get("loss_masking", "none") != "none" and visible_ratio_value is not None:
        loss_visibility_ratio = visible_ratio_value

    render_losses = loss_manager.compute_render_loss(
        pred_render, target_render,
        cov=cov, mu=mu,
        view_params=view_params,
        cov_target=cov_target_filtered,  # 🔥 필터링된 복사본 전달
        F=F,  # 🔥 NEW: det(F) 바리어 손실용
        surface_mask=surface_mask,  # 🔥 CRITICAL: Pass surface mask to prevent gradient dilution
        opacity=opacity,  # 🔥 NEW: Per-Gaussian opacity for shrinkage regularization
        visibility_ratio=loss_visibility_ratio
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
    
    if not torch.isfinite(loss_render):
        raise FloatingPointError("Render loss produced NaN/Inf before backward")
    
    print(f"├─ Backward pass starting...")
    loss_render.backward()
    
    if visible_only and visible_mask is not None:
        vis = visible_mask.to(mu.device)
        if x.grad is not None and x.grad.shape[0] == vis.shape[0]:
            x.grad *= vis.view(-1, 1)
        if F.grad is not None and F.grad.shape[0] == vis.shape[0]:
            F.grad *= vis.view(-1, 1, 1)
        print(f"│  🎯 Gradients zeroed for {vis.numel() - vis.sum().item():,} non-visible particles")
        if grad_focus_cfg.get("loss_masking", "none") == "edge" and surface_mask is not None:
            mask_loss = vis.bool()
            render_losses['visible_mask_ratio'] = float(mask_loss.float().mean().item())
            render_losses['visible_mask_count'] = int(mask_loss.sum().item())
            render_losses['visible_mask_mode'] = "edge"
    
    nan_sources = []
    if F.grad is None or x.grad is None:
        raise RuntimeError("Render backward produced no gradients")
    
    if not torch.isfinite(F.grad).all():
        nan_sources.append("F.grad")
    if not torch.isfinite(x.grad).all():
        nan_sources.append("x.grad")
    
    if nan_sources:
        raise FloatingPointError(f"Render gradients invalid: {', '.join(nan_sources)}")
    
    print(f"└─ ✅ Backward completed, gradients finite")
    
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

    # 🔍 DEBUG: Check F.grad magnitude
    F_grad_norm = torch.linalg.norm(F.grad).item()
    print(f"[DEBUG] F.grad magnitude at extraction: {F_grad_norm:.12e}")
    print(f"[DEBUG] F.grad range: [{F.grad.min().item():.6e}, {F.grad.max().item():.6e}]")
    print(f"[DEBUG] F.requires_grad: {F.requires_grad}")

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
