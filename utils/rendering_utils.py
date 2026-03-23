"""
Rendering Utilities - 3D Gaussian Splatting & Target Rendering

Handles renderer setup, target rendering, and loss computation.
"""

import numpy as np
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


# ============================================================================
# Renderer Setup
# ============================================================================

def setup_renderer(
    cam_cfg: Dict,
    render_cfg: Dict,
    training_mode: bool = False
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
# Rendering Inputs
# ============================================================================

def prepare_rendering_inputs(
    mu: torch.Tensor,
    result: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list
) -> torch.Tensor:
    """
    Prepare RGB colors for rendering using Phong shading.

    Args:
        mu: Positions (M, 3)
        result: Upsampling result with normals
        campos: Camera position
        render_cfg: Rendering configuration
        particle_color: Base albedo color

    Returns:
        rgb: (M, 3) shaded colors
    """
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

    return torch.from_numpy(rgb_np).to(mu.device)


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
    current_episode: int = 0
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
    """
    Upsample current simulation state for differentiable rendering.

    Args:
        pc: Point cloud from MPM simulation
        rs_full: Upsampling configuration
        ema_state: EMA state (updated in-place)
        seed: Random seed
        cov_module: Optional learnable covariance module
        export_stages: Whether to export stage visualizations
        external_levelset: Pre-computed level set
        current_episode: Current episode number

    Returns:
        Tuple of (mu, cov, result)
    """
    try:
        x = pc.get_positions_torch(requires_grad=True)
        F = pc.get_def_grads_total_torch(requires_grad=True)
    except AttributeError:
        print("      ⚠️  PyTorch bindings unavailable")
        return None, None, ema_state

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
        current_episode=current_episode
    )

    mu = result["points"]
    cov = result.get("cov_pred", result["cov"])

    return mu, cov, result


# ============================================================================
# Target Rendering
# ============================================================================

def generate_target_render(
    target_positions: np.ndarray,
    rs_full: Dict,
    renderer: Any,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    training_resolution_scale: float = 0.5,
    target_mesh_path: Optional[str] = None,
    cam_cfg: Optional[Dict] = None,
) -> Optional[torch.Tensor]:
    """
    Generate target silhouette (alpha) by mesh rasterization.

    Uses pyrender to render the target mesh directly, producing a clean
    binary silhouette. Falls back to depth-based alpha if mesh is available.

    Args:
        target_positions: (N, 3) bunny particle positions (used as fallback)
        rs_full: Upsampling config (unused with mesh rasterization)
        renderer: GSRenderer3DGS instance (used for resolution info)
        campos: Camera position
        render_cfg: Render config dict
        particle_color: Base albedo color
        training_resolution_scale: Resolution scale
        target_mesh_path: Path to target .obj mesh
        cam_cfg: Camera config dict for building projection

    Returns:
        alpha: (H, W) float32 target alpha tensor, or None if failed
    """
    N = target_positions.shape[0]
    print(f"[Target] Generating target alpha from mesh rasterization ({N:,} particles)...")

    # Try mesh rasterization first
    if target_mesh_path is not None and cam_cfg is not None:
        alpha = _render_mesh_silhouette(target_mesh_path, cam_cfg, render_cfg)
        if alpha is not None:
            print(f"  alpha shape={alpha.shape}, range=[{alpha.min():.3f}, {alpha.max():.3f}]")
            return alpha
        print("[Target] Mesh rasterization failed, falling back to Gaussian render")

    # Fallback: Gaussian render with large sigma
    print("[Target] Using Gaussian fallback with sigma0=1.0")
    x_t = torch.from_numpy(target_positions.astype(np.float32))
    F_id = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)

    # Override sigma0 to be large enough for visibility
    rs_target = dict(rs_full)
    if 'upsample' in rs_target:
        rs_target['upsample'] = dict(rs_target['upsample'])
        if 'covariance' in rs_target['upsample']:
            rs_target['upsample']['covariance'] = dict(rs_target['upsample']['covariance'])
            rs_target['upsample']['covariance']['sigma0'] = 1.0

    with torch.no_grad():
        result = upsample(x_t, F_id, cfg=rs_target, seed=42, return_torch=True)

    mu  = result.get('points')
    cov = result.get('cov')
    if mu is None or cov is None:
        print("[Target] upsample returned None — target render failed")
        return None

    rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)

    with torch.no_grad():
        pred = renderer.render(mu, cov, rgb=rgb, prefer_cov_precomp=True, return_torch=True)

    alpha = pred.get('alpha')
    if alpha is None:
        print("[Target] renderer returned no alpha")
        return None

    if alpha.dim() == 3:
        alpha = alpha[0]

    alpha = alpha.detach().cpu()
    print(f"  alpha shape={alpha.shape}, range=[{alpha.min():.3f}, {alpha.max():.3f}]")
    return alpha


def _render_mesh_silhouette(
    mesh_path: str,
    cam_cfg: Dict,
    render_cfg: Dict,
) -> Optional[torch.Tensor]:
    """Render mesh silhouette using pyrender (offscreen)."""
    try:
        import trimesh
        import pyrender
        import os
        os.environ['PYOPENGL_PLATFORM'] = 'egl'  # headless rendering

        mesh = trimesh.load(mesh_path, force='mesh')
        pr_mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
        scene.add(pr_mesh)

        # Camera setup from config
        W = int(cam_cfg.get('width', 1920))
        H = int(cam_cfg.get('height', 1080))
        fx = float(cam_cfg.get('fx', 712.5))
        fy = float(cam_cfg.get('fy', 712.5))
        cx = float(cam_cfg.get('cx', W / 2))
        cy = float(cam_cfg.get('cy', H / 2))

        # Training resolution
        scale = float(render_cfg.get('training_resolution_scale', 0.5))
        if scale < 1.0:
            W_r = int(W * scale)
            H_r = int(H * scale)
            fx_r = fx * scale
            fy_r = fy * scale
            cx_r = cx * scale
            cy_r = cy * scale
        else:
            W_r, H_r = W, H
            fx_r, fy_r, cx_r, cy_r = fx, fy, cx, cy

        camera = pyrender.IntrinsicsCamera(fx=fx_r, fy=fy_r, cx=cx_r, cy=cy_r,
                                            znear=float(cam_cfg.get('znear', 0.01)),
                                            zfar=float(cam_cfg.get('zfar', 100.0)))

        # Camera pose: lookat → 4x4 extrinsic
        lookat = cam_cfg.get('lookat', {})
        eye = np.array(lookat.get('eye', [20, -25, 12.5]), dtype=np.float64)
        target = np.array(lookat.get('target', [0, 0, 0]), dtype=np.float64)
        up = np.array(lookat.get('up', [0, 0, 1]), dtype=np.float64)

        # Build camera-to-world matrix (OpenGL convention: -Z forward)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = true_up
        cam_pose[:3, 2] = -forward  # OpenGL: camera looks along -Z
        cam_pose[:3, 3] = eye

        scene.add(camera, pose=cam_pose)

        # Add a light so depth buffer works
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=cam_pose)

        r = pyrender.OffscreenRenderer(W_r, H_r)
        _, depth = r.render(scene)
        r.delete()

        # depth > 0 means something was rendered → alpha = 1
        alpha = (depth > 0).astype(np.float32)
        return torch.from_numpy(alpha)

    except Exception as e:
        print(f"[Target] pyrender mesh rasterization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


__all__ = [
    'setup_renderer',
    'prepare_rendering_inputs',
    'upsample_current_state',
    'generate_target_render',
]
