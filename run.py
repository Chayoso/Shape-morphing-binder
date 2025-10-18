"""
run.py - [CVPR 2026] PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing

This module implements an end-to-end training pipeline that combines:
1. Physics-based Material Point Method (MPM) simulation
2. Differentiable surface synthesis with runtime upsampling
3. 3D Gaussian Splatting rendering
4. Multi-modal rendering losses (silhouette, depth, edge alignment, covariance)

Architecture Overview:
──────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────┐
│                        Input Mesh                               │
│                  (Sparse particle cloud)                        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MPM Simulation (C++)                         │
│  • Forward dynamics: x(t), F(t) for each timestep               │
│  • Backward gradients: ∂L_physics/∂x, ∂L_physics/∂F             │
│  • Adam optimization over control timesteps                     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│          Runtime Surface Synthesis (Differentiable)             │
│  • Surface detection: PCA-based normal estimation               │
│  • Upsampling: x_low (N) → μ_high (M), M >> N                   │
│  • Covariance: Σ = σ₀² F·Fᵀ (from deformation gradients)        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              3D Gaussian Splatting Renderer                     │
│  • Input: (μ, Σ, RGB) - positions, covariances, colors          │
│  • Output: {image, alpha, depth, normal_map}                    │
│  • Differentiable rasterization via PyTorch                     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Rendering Loss Manager                       │
│  L_render = w_α·L_α + w_edge·L_edge + w_cov·L_cov_align         │
│                                                                 │
│  • L_α: Silhouette supervision (alpha channel)                  │
│  • L_edge: Edge alignment (2D projection of Σ vs silhouette)    │
│  • L_cov_align: Spectral align (Σ_pred vs. Σ_target)            │
│  • Σ_target = curvature-based anisotropic target covariance     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              Gradient Backpropagation Chain                     │
│                                                                 │
│  L_render → ∂L/∂Σ → ∂L/∂F → ∂L/∂x                               │
│             (cov) (def grad) (position)                         │
│                                                                 │
│  These gradients are injected back to MPM simulation:           │
│  L_total = L_physics + λ·L_render                               │
└─────────────────────────────────────────────────────────────────┘

Training Loop (E2E Interleaved Mode):
──────────────────────────────────────────────────────────────────────
For each episode:
  Setup: Create computation graph with T timesteps
  
  For each pass (default: 3 passes):
    
    Phase 1: Inject Render Gradients
      • If not first pass: inject ∂L_render/∂F, ∂L_render/∂x from previous pass
      • C++ backend combines: ∂L_total = ∂L_physics + ∂L_render
    
    Phase 2: Physics Optimization
      • For each control timestep t:
        - Forward: simulate dynamics x(t) → x(t+1)
        - Compute: L_physics = ||x_final - x_target||²
        - Backward: compute ∂L_total/∂x(t)
        - Update: Adam step on control forces
    
    Phase 3: Render Loss Computation
      • Upsample final state: (x_low, F_low) → (μ_high, Σ_high)
      • Render: (μ, Σ, RGB) → {image, alpha, depth}
      • Compare with target render: compute L_render
      • Backprop: L_render.backward() to get ∂L/∂F, ∂L/∂x
      • Store gradients for next pass
    
    Phase 4: Visualization (last pass only)
      • Save: rendered images, normal maps, point clouds
      • Export: NPZ (Gaussians), PLY (mesh), PNG (images)
  
  Promote final state to next episode

Key Features:
──────────────────────────────────────────────────────────────────────
✅ Differentiable Pipeline: End-to-end gradient flow from pixels to physics
✅ Curvature-Aware Supervision: Σ★ target derived from surface geometry
✅ Silhouette Alignment: Edge loss guides 2D projection of 3D Gaussians
✅ Multi-Pass Training: Iterative refinement within each episode
✅ Modular Design: Clean separation of physics, synthesis, rendering, loss

Physics Backend:
  • C++ MPM simulation with PyTorch bindings
  • Supports elastic materials (Neo-Hookean)
  • Gradient injection for render supervision

Rendering Backend:
  • 3D Gaussian Splatting (3DGS)
  • Differentiable rasterization
  • Supports: RGB, alpha, depth, normal maps

Surface Synthesis:
  • Hybrid FAISS KNN for efficiency
  • PCA-based surface detection
  • Density equalization and smoothing
  • Optional normal smoothing

Loss Components:
  • w_alpha (0.3-0.7): Silhouette matching
  • w_edge (0.05-0.2): Edge alignment
  • w_cov_align (0.05-0.3): Covariance spectral loss
  • w_depth (0.1): Depth consistency (optional)
  • w_cov_reg (0.01): Covariance regularization

Configuration:
──────────────────────────────────────────────────────────────────────
YAML config structure:
  input_mesh_path: path to initial shape (.obj)
  target_mesh_path: path to target shape (.obj)
  simulation: {grid_dx, lam, mu, density, dt, drag, ...}
  optimization: {num_timesteps, control_stride, max_gd_iters, ...}
  render: {particle_color, lighting, bg}
  camera: {position, target, fov, resolution}
  sampling: {runtime_surface: {M, k_surface, sigma0, ...}}
  optimization.loss: {enabled, w_alpha, w_edge, w_cov_align, ...}

Usage:
──────────────────────────────────────────────────────────────────────
# E2E mode with rendering supervision:
python run.py -c config.yaml --e2e

# Physics-only mode (no rendering loss):
python run.py -c config.yaml

# With PNG export:
python run.py -c config.yaml --e2e --png

Output Structure:
──────────────────────────────────────────────────────────────────────
output/
├── target/
│   ├── target_image.png         # Target render (RGB)
│   ├── target_alpha.png         # Target silhouette
│   ├── target_depth.png         # Target depth (16-bit)
│   ├── target_normal.png        # Target normals
│   └── target_render.npz        # All target data
├── ep000/
│   ├── ep000_render.png         # Final render
│   ├── ep000_alpha.png          # Final silhouette
│   ├── ep000_depth.png          # Final depth
│   ├── ep000_normal.png         # Final normals
│   ├── ep000_gaussians.npz      # (μ, Σ, RGB) for 3DGS
│   ├── ep000_surface_50000.ply  # Point cloud
│   ├── ep000_comparison.png     # Before/after visualization
│   ├── ep000_axis_hist.png      # Distribution analysis
│   └── ep000_summary.json       # Loss metrics, statistics
├── ep001/
│   └── ...
└── ...

Dependencies:
──────────────────────────────────────────────────────────────────────
- diffmpm_bindings: C++ MPM simulation (custom)
- sampling.runtime_surface: Differentiable upsampling (custom)
- renderer.renderer: 3D Gaussian Splatting (custom)
- loss: E2E loss manager with curvature support (custom)
- torch, numpy, imageio, PyYAML

Authors: DiffMPM Team
License: MIT (or your license)
Version: 2.0 (E2E + Silhouette + Curvature)
Last Updated: 2025-01-XX
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch

# Image I/O utilities
try:
    import imageio.v2 as iio
    def _save_png(path, img):
        iio.imwrite(str(path), img)
except Exception:
    from PIL import Image
    def _save_png(path, img):
        Image.fromarray(img).save(str(path))


def _save_depth16(path, depth_meters):
    """Save depth map in 16-bit format (millimeters)."""
    dmm = np.clip(np.nan_to_num(depth_meters, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535)
    dmm = dmm.astype(np.uint16)
    try:
        iio.imwrite(str(path), dmm)
    except Exception:
        from PIL import Image as _Image
        _Image.fromarray(dmm).save(str(path))


# Import core modules
try:
    import diffmpm_bindings
    BINDINGS_AVAILABLE = True
except Exception:
    diffmpm_bindings = None
    BINDINGS_AVAILABLE = False

# ============================================================================
# Sampling Import
# ============================================================================
try:
    from sampling import (
        default_cfg,                   
        synthesize_runtime_surface,
        save_ply_xyz,
        save_gaussians_npz,
        save_comparison_png,
        save_axis_hist_png,
    )
    DIFFERENTIABLE_MODE = True
except ImportError as e:
    print(f"⚠️ RuntimeSurface not available: {e}")
    DIFFERENTIABLE_MODE = False

from renderer import (
    GSRenderer3DGS,
    make_matrices_from_yaml,
    compute_shading,
    composite_with_background,
    LightConfig,
    RenderConfig,
)

try:
    from loss import E2ELossManager, estimate_curvature, create_target_covariance
except Exception:
    print("[WARN] loss.py not found. E2E training disabled.")
    E2ELossManager = None
    estimate_curvature = None
    create_target_covariance = None


# ============================================================================
# Constants
# ============================================================================
DEFAULT_PARTICLE_COLOR = [0.7, 0.7, 0.7]
DEFAULT_BG_COLOR = [1.0, 1.0, 1.0]


# ============================================================================
# Utility Functions
# ============================================================================
def _np(x):
    """Convert to numpy array."""
    return np.asarray(x)


def _pick_timesteps(num_layers: int, num_frames: int, schedule: str = "last_n"):
    """Select timestep indices for visualization."""
    num_frames = int(max(0, num_frames))
    if num_frames == 0 or num_layers <= 0:
        return []
    num_frames = min(num_frames, num_layers)
    
    if schedule == "uniform":
        if num_frames == 1:
            return [num_layers - 1]
        xs = np.linspace(0, num_layers - 1, num_frames)
        return sorted(set(int(round(v)) for v in xs))
    
    # last_n schedule
    start = max(0, num_layers - num_frames)
    return list(range(start, num_layers))


# ============================================================================
# Configuration Loading
# ============================================================================
def load_config(config_path: str) -> Tuple[Dict, Path]:
    """Load configuration from YAML file."""
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    return cfg, config_path.parent


def validate_config(cfg: Dict) -> None:
    """Validate required configuration fields."""
    required = ["input_mesh_path", "target_mesh_path", "simulation", "optimization"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config field: {key}")


def build_opt_input(cfg: Dict) -> Any:
    """Build OptInput object from configuration."""
    if not BINDINGS_AVAILABLE:
        raise RuntimeError("C++ diffmpm_bindings are required.")
    
    sim = cfg["simulation"]
    run = cfg["optimization"]
    
    opt = diffmpm_bindings.OptInput()
    opt.mpm_input_mesh_path = cfg["input_mesh_path"]
    opt.mpm_target_mesh_path = cfg["target_mesh_path"]
    opt.grid_dx = float(sim["grid_dx"])
    opt.grid_min_point = sim["grid_min_point"]
    opt.grid_max_point = sim["grid_max_point"]
    opt.points_per_cell_cuberoot = int(sim.get("points_per_cell_cuberoot", 2))
    opt.lam = float(sim["lam"])
    opt.mu = float(sim["mu"])
    opt.p_density = float(sim["density"])
    opt.dt = float(sim["dt"])
    opt.drag = float(sim["drag"])
    opt.f_ext = sim["external_force"]
    opt.smoothing_factor = float(sim["smoothing_factor"])
    opt.num_animations = int(run["num_animations"])
    opt.num_timesteps = int(run["num_timesteps"])
    opt.control_stride = int(run["control_stride"])
    opt.max_gd_iters = int(run["max_gd_iters"])
    opt.max_ls_iters = int(run["max_ls_iters"])
    opt.initial_alpha = float(run["initial_alpha"])
    opt.gd_tol = float(run["gd_tol"])
    opt.current_episodes = 0
    
    return opt


# ============================================================================
# Initialization
# ============================================================================
def initialize_point_clouds(opt: Any) -> Tuple[Any, Any]:
    """Load and initialize input and target point clouds."""
    input_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_input_mesh_path, opt)
    target_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_target_mesh_path, opt)
    return input_pc, target_pc


def initialize_grids(opt: Any) -> Tuple[Any, Any]:
    """Create and initialize simulation grids."""
    grid_dims = [
        int((opt.grid_max_point[i] - opt.grid_min_point[i]) / opt.grid_dx) + 1 
        for i in range(3)
    ]
    
    input_grid = diffmpm_bindings.Grid(
        grid_dims[0], grid_dims[1], grid_dims[2], 
        opt.grid_dx, opt.grid_min_point
    )
    target_grid = diffmpm_bindings.Grid(
        grid_dims[0], grid_dims[1], grid_dims[2], 
        opt.grid_dx, opt.grid_min_point
    )
    
    return input_grid, target_grid


def initialize_comp_graph(
    input_pc: Any, 
    input_grid: Any, 
    target_grid: Any
) -> Any:
    """Create computation graph."""
    return diffmpm_bindings.CompGraph(input_pc, input_grid, target_grid)


# ============================================================================
# Renderer Setup
# ============================================================================
def setup_renderer(
    cam_cfg: Dict, 
    render_cfg: Dict
) -> Tuple[Optional[Any], Dict]:
    """Initialize 3D Gaussian Splatting renderer."""
    try:
        W, H, tanfovx, tanfovy, view_T, proj_T, campos = make_matrices_from_yaml(cam_cfg)
        
        bg = render_cfg.get("bg", DEFAULT_BG_COLOR)
        
        renderer = GSRenderer3DGS(
            W, H, tanfovx, tanfovy, view_T, proj_T, campos,
            bg=tuple(bg), 
            sh_degree=0, 
            scale_modifier=1.0, 
            prefiltered=False, 
            debug=False, 
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
# Target Render Creation
# ============================================================================
def extract_target_point_cloud(target_pc: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract positions and deformation gradients from target point cloud."""
    try:
        x_tgt = target_pc.get_positions_torch(requires_grad=False)
        F_tgt = target_pc.get_def_grads_total_torch(requires_grad=False)
    except AttributeError:
        x_tgt = _np(target_pc.get_positions())
        F_tgt = _np(target_pc.get_def_grads_total())
    return x_tgt, F_tgt


def upsample_target(
    x_tgt: np.ndarray, 
    F_tgt: np.ndarray, 
    rs: Dict
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Upsample target point cloud to create dense surface."""
    result_tgt = synthesize_runtime_surface(
        x_tgt, F_tgt, rs,
        seed=9999,
        differentiable=False,
        return_torch=False
    )
    
    mu_tgt = result_tgt["points"]
    cov_tgt = result_tgt["cov"]
    nrm_tgt = result_tgt.get("normals")
    
    return mu_tgt, cov_tgt, nrm_tgt


def compute_target_covariance_star(
    mu_tgt: np.ndarray, 
    nrm_tgt: Optional[np.ndarray]
) -> Optional[torch.Tensor]:
    """Compute curvature-based target covariance Σ★."""
    if estimate_curvature is None or create_target_covariance is None or nrm_tgt is None:
        return None
    
    print("  Computing target covariance Σ★ from curvature...")
    try:
        mu_tgt_torch = torch.from_numpy(mu_tgt).cuda()
        nrm_tgt_torch = torch.from_numpy(nrm_tgt).cuda()
        
        # Estimate curvatures
        curvatures = estimate_curvature(mu_tgt_torch, nrm_tgt_torch, k=16)
        
        # Create target covariance
        cov_target_star = create_target_covariance(
            mu_tgt_torch, 
            nrm_tgt_torch, 
            curvatures,
            base_scale=0.02,
            aniso_factor=2.0
        )
        
        print(f"    ✅ Target Σ★ created: mean curvature = {curvatures.mean(dim=0).cpu().numpy()}")
        return cov_target_star
    except Exception as e:
        print(f"    ⚠️ Σ★ creation failed: {e}")
        return None


def render_target(
    renderer: Any,
    mu_tgt: np.ndarray,
    cov_tgt: np.ndarray,
    nrm_tgt: Optional[np.ndarray],
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list
) -> Dict:
    """Render target point cloud."""
    if nrm_tgt is None:
        nrm_tgt = np.zeros_like(mu_tgt)
    
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
        mu_tgt, cov_tgt, rgb=rgb_tgt,
        normals=nrm_tgt,  
        prefer_cov_precomp=True,
        return_torch=False,
        render_normal_map=True  
    )
    
    return out_tgt


def normalize_render_outputs(out_tgt: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:  # ✅ 반환 타입 변경
    """Normalize alpha, depth, and normal map shapes."""
    alpha_np = out_tgt['alpha']
    depth_np = out_tgt['depth']
    normal_map_np = out_tgt.get('normal_map') 
    
    # Normalize alpha
    if alpha_np.ndim == 3:
        if alpha_np.shape[0] in (1, 3, 4):
            alpha_np = alpha_np[0] if alpha_np.shape[0] == 1 else alpha_np.mean(axis=0)
        elif alpha_np.shape[-1] in (1, 3, 4):
            alpha_np = alpha_np[..., 0] if alpha_np.shape[-1] == 1 else alpha_np.mean(axis=-1)
    
    # Normalize depth
    if depth_np is not None and depth_np.ndim == 3:
        if depth_np.shape[0] == 1:
            depth_np = depth_np[0]
        elif depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
    
    # Normalize normal map
    if normal_map_np is not None:
        if normal_map_np.ndim == 3 and normal_map_np.shape[0] == 3:
            normal_map_np = normal_map_np.transpose(1, 2, 0)
    
    return out_tgt['image'], alpha_np, depth_np, normal_map_np  


def save_target_renders(
    save_dir: Path,
    image_np: np.ndarray,
    alpha_np: np.ndarray,
    depth_np: Optional[np.ndarray],
    normal_map_np: Optional[np.ndarray],
    renderer: Any
):
    """Save target render images."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ
    np.savez(
        save_dir / "target_render.npz",
        image=image_np,
        alpha=alpha_np,
        depth=depth_np if depth_np is not None else np.zeros((renderer.height, renderer.width)),
        normal_map=normal_map_np if normal_map_np is not None else np.zeros((renderer.height, renderer.width, 3))
    )
    
    # Save PNGs
    _save_png(save_dir / "target_image.png", 
             (np.clip(image_np, 0, 1) * 255).astype(np.uint8))
    
    if alpha_np is not None:
        _save_png(save_dir / "target_alpha.png", 
                 (np.clip(alpha_np, 0, 1) * 255).astype(np.uint8))
    
    if depth_np is not None:
        _save_depth16(save_dir / "target_depth.png", depth_np)
    
    # Save target normal map with validation
    if normal_map_np is not None:
        if normal_map_np.ndim == 3 and normal_map_np.shape[-1] == 3:
            _save_png(save_dir / "target_normal.png",
                     (np.clip(normal_map_np, 0, 1) * 255).astype(np.uint8))
        else:
            print(f"[WARN] Target normal map has invalid shape: {normal_map_np.shape}, skipping")


def create_target_render(
    target_pc: Any,
    renderer: Any,
    rs: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    save_dir: Path,
    view_T: np.ndarray
) -> Dict:
    """Create target render for supervision."""
    print("  Extracting target point cloud...")
    x_tgt, F_tgt = extract_target_point_cloud(target_pc)
    
    print("  Upsampling target...")
    mu_tgt, cov_tgt, nrm_tgt = upsample_target(x_tgt, F_tgt, rs)
    print(f"  Target upsampled: {len(mu_tgt)} points")
    
    cov_target_star = compute_target_covariance_star(mu_tgt, nrm_tgt)
    
    print("  Rendering target...")
    out_tgt = render_target(renderer, mu_tgt, cov_tgt, nrm_tgt, campos, render_cfg, particle_color)
    
    # Normalize target render outputs
    image_np, alpha_np, depth_np, normal_map_np = normalize_render_outputs(out_tgt)
    
    # Save target renders
    save_target_renders(save_dir, image_np, alpha_np, depth_np, normal_map_np, renderer)
    
    target_dict = {
        'image': torch.from_numpy(image_np).cuda(),
        'alpha': torch.from_numpy(alpha_np).cuda(),
        'depth': torch.from_numpy(depth_np).cuda() if depth_np is not None else None,
        'cov_target': cov_target_star.cuda() if cov_target_star is not None else None,
    }
    
    print(f"  Target shapes:")
    print(f"    - image: {target_dict['image'].shape}")
    print(f"    - alpha: {target_dict['alpha'].shape}")
    print(f"    - depth: {target_dict['depth'].shape if target_dict['depth'] is not None else None}")
    if cov_target_star is not None:
        print(f"    - cov_target: {cov_target_star.shape}")
    if normal_map_np is not None:
        print(f"    - normal_map: {normal_map_np.shape}")  
    
    return target_dict

# ============================================================================
# Physics Optimization
# ============================================================================
def run_physics_optimization(
    cg: Any,
    opt: Any,
    num_timesteps: int,
    control_stride: int,
    ep: int,
    pass_idx: int = 0
) -> float:
    """Run physics optimization over control timesteps."""
    control_timesteps = list(range(0, num_timesteps - 1, control_stride))
    
    print(f"\n[Physics] Pass {pass_idx+1} - Optimizing {len(control_timesteps)} timesteps (stride={control_stride})")
    
    loss_physics = 0.0
    
    for i, t in enumerate(control_timesteps):
        print(f"├─ Timestep {t}/{num_timesteps-1} ({i+1}/{len(control_timesteps)})")
        
        # Forward pass
        cg.compute_forward_pass(t, ep)
        
        # Physics loss
        try:
            loss_physics = cg.end_layer_mass_loss()
            print(f"│  ├─ Physics loss: {loss_physics:.2f}")
        except Exception as e:
            print(f"│  ├─ [ERROR] Physics loss failed: {e}")
            loss_physics = 0.0
        
        # Check if render gradients available
        has_render_grads = cg.has_render_gradients()
        if has_render_grads:
            # Get gradient info before backward (C++ might print here)
            print(f"│  ├─ [Render Grads] Injecting to layer {num_timesteps-1}")
        
        # Backward pass
        cg.compute_backward_pass(t)
        
        # Adam update
        initial_loss = loss_physics
        cg.optimize_single_timestep(
            t, 
            max_gd_iters=opt.max_gd_iters, 
            current_episode=ep,
            initial_alpha=opt.initial_alpha,
            max_line_search_iters=opt.max_ls_iters
        )
        
        # Get final loss
        try:
            final_loss = cg.end_layer_mass_loss()
            reduction = initial_loss - final_loss
            print(f"│  ├─ Optimization: Δloss = {reduction:.2f}")
            print(f"│  └─ Final loss: {final_loss:.2f} {'✅' if has_render_grads else '(physics only)'}")
        except:
            print(f"│  └─ Optimization completed")
    
    print(f"└─ [Physics] Pass {pass_idx+1} completed\n")
    return loss_physics


# ============================================================================
# Render Loss Computation
# ============================================================================
def upsample_current_state(
    pc: Any,
    rs_full: Dict,
    ema_state: Dict,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Upsample current state for rendering."""
    try:
        x = pc.get_positions_torch(requires_grad=True)
        F = pc.get_def_grads_total_torch(requires_grad=True)
    except AttributeError:
        print("      ⚠️  PyTorch bindings unavailable")
        return None, None, ema_state
    
    result = synthesize_runtime_surface(
        x, F, rs_full,
        ema_state=ema_state,
        seed=seed,
        differentiable=True,
        return_torch=True
    )
    
    mu = result["points"]
    cov = result["cov"]
    ema_state = result["state"]
    
    return mu, cov, ema_state


def prepare_rendering_inputs(
    mu: torch.Tensor,
    result: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list
) -> torch.Tensor:
    """Prepare RGB colors for rendering."""
    mu_np = mu.detach().cpu().numpy()
    nrm_np = result.get("normals")
    
    if nrm_np is not None and torch.is_tensor(nrm_np):
        nrm_np = nrm_np.detach().cpu().numpy()
    elif nrm_np is None:
        nrm_np = np.zeros_like(mu_np)
    
    rgb_np = compute_shading(
        mu_np, nrm_np,
        camera_pos=campos,
        light_cfg=render_cfg.get("lighting", {}),
        albedo_color=particle_color,
        model="phong"
    )
    
    rgb = torch.from_numpy(rgb_np).to(mu.device)
    return rgb


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
    seed: int
) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor]]: 
    """Compute render loss for current pass."""
    
    torch.cuda.empty_cache()
    
    # Get final state
    pc = cg.get_point_cloud(num_timesteps - 1)
    
    try:
        x = pc.get_positions_torch(requires_grad=True)
        F = pc.get_def_grads_total_torch(requires_grad=True)
    except AttributeError:
        print("   ⚠️ PyTorch bindings unavailable")
        return None, None, None
    
    # Upsample
    result = synthesize_runtime_surface(
        x, F, rs_full,
        ema_state=ema_state,
        seed=seed,
        differentiable=True,
        return_torch=True
    )
    
    mu = result["points"]
    cov = result["cov"]
    ema_state = result["state"]
    
    if mu is None:
        return None, None, None
    
    print(f"├─ Upsampled: {len(mu)} points")
    
    # Prepare rendering
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
    
    # Print losses with tree structure
    print(f"├─ Render loss: {loss_render.item():.6f}")
    for key in ['loss_alpha', 'loss_edge', 'loss_cov_align']:
        if key in render_losses:
            val = render_losses[key]
            if torch.is_tensor(val):
                print(f"│  ├─ {key}: {val.item():.6f}")
    
    # Backward
    loss_render.backward()
    
    return ema_state, F, x


def extract_render_gradients(F: torch.Tensor, x: torch.Tensor) -> Optional[Dict]:
    """Extract and process render gradients."""
    if F.grad is None:
        return None
    
    # Clean gradients
    F.grad = torch.nan_to_num(F.grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad_norm = torch.norm(F.grad).item()
    
    # Gradient clipping
    if grad_norm > 10.0:
        F.grad = F.grad * (10.0 / grad_norm)
        print(f"   ⚠️ Gradient clipped: {grad_norm:.2e} → 10.0")
    
    dLdF = F.grad.detach().cpu().numpy().astype(np.float32)
    dLdx = x.grad.detach().cpu().numpy().astype(np.float32) if x.grad is not None else np.zeros_like(x.detach().cpu().numpy(), dtype=np.float32)
    
    print(f"├─ ||∂L_render/∂F|| = {grad_norm:.6e}")
    
    return {
        'dLdF': dLdF,
        'dLdx': dLdx,
    }


# ============================================================================
# Visualization
# ============================================================================
def visualize_episode(
    ep: int,
    out_dir: Path,
    cg: Any,
    num_timesteps: int,
    rs_full: Dict,
    ema_state: Dict,
    renderer: Any,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    png_enabled: bool,
    tgt: np.ndarray,
    loss_physics: float,
    seed: int
):
    """Save visualization outputs for an episode."""
    print(f"\n  [Visualization] Saving episode {ep+1} outputs...")
    ep_dir = out_dir / f"ep{ep:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    
    # Get final state
    pc = cg.get_point_cloud(num_timesteps - 1)
    try:
        x = pc.get_positions_torch(requires_grad=False)
        F = pc.get_def_grads_total_torch(requires_grad=False)
    except AttributeError:
        x = _np(pc.get_positions())
        F = _np(pc.get_def_grads_total())
    
    # Upsample
    result = synthesize_runtime_surface(
        x, F, rs_full,
        ema_state=ema_state,
        seed=seed,
        differentiable=False,
        return_torch=False
    )
    
    mu_np = result["points"]
    cov_np = result["cov"]
    nrm_np = result.get("normals")
    
    if nrm_np is None:
        nrm_np = np.zeros_like(mu_np)
    
    # Compute shading
    rgb_np = compute_shading(
        mu_np, nrm_np,
        camera_pos=campos,
        light_cfg=render_cfg.get("lighting", {}),
        albedo_color=particle_color,
        model="phong"
    )
    
    # Render main image 
    out_final = renderer.render(
        mu_np, cov_np, rgb=rgb_np,
        normals=nrm_np,  
        prefer_cov_precomp=True,
        return_torch=False,
        render_normal_map=True 
    )
    
    # Save images (including normal map)
    save_episode_images(ep, ep_dir, out_final, renderer)
    
    # Save comparison PNGs
    if png_enabled:
        save_episode_comparisons(ep, ep_dir, x, mu_np, tgt, rs_full)
    
    # Save summary
    save_episode_summary(ep, ep_dir, F, mu_np, loss_physics)
    
    # Save data files
    save_episode_data(ep, ep_dir, mu_np, cov_np, particle_color)
    
    print(f"  ✅ All outputs saved to {ep_dir}")


def save_episode_images(
    ep: int,
    ep_dir: Path,
    out_final: Dict,
    renderer: Any
):
    """Save rendered images for episode."""
    img_np = out_final['image']
    alpha_np = out_final['alpha']
    depth_np = out_final['depth']
    normal_map_np = out_final.get('normal_map')
    
    # Normalize alpha
    if alpha_np is not None and alpha_np.ndim == 3:
        if alpha_np.shape[0] in (1, 3, 4):
            alpha_np = alpha_np[0] if alpha_np.shape[0] == 1 else alpha_np.mean(axis=0)
        elif alpha_np.shape[-1] in (1, 3, 4):
            alpha_np = alpha_np[..., 0] if alpha_np.shape[-1] == 1 else alpha_np.mean(axis=-1)
    
    # Normalize depth
    if depth_np is not None and depth_np.ndim == 3:
        if depth_np.shape[0] == 1:
            depth_np = depth_np[0]
        elif depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
    
    # Normalize normal map with validation
    if normal_map_np is not None:
        if normal_map_np.ndim == 3:
            if normal_map_np.shape[-1] == 3:
                # Already (H, W, 3), perfect
                pass
            elif normal_map_np.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                normal_map_np = normal_map_np.transpose(1, 2, 0)
            else:
                print(f"[WARN] Unexpected normal map shape: {normal_map_np.shape}")
                normal_map_np = None
        elif normal_map_np.ndim == 2:
            # Grayscale (H, W) - should not happen, but handle it
            print(f"[WARN] Normal map is 2D: {normal_map_np.shape}, skipping")
            normal_map_np = None
        else:
            print(f"[WARN] Invalid normal map shape: {normal_map_np.shape}")
            normal_map_np = None
    
    # Save main render
    _save_png(ep_dir / f"ep{ep:03d}_render.png", 
             (np.clip(img_np, 0, 1) * 255).astype(np.uint8))
    
    # Save alpha
    if alpha_np is not None:
        _save_png(ep_dir / f"ep{ep:03d}_alpha.png", 
                 (np.clip(alpha_np, 0, 1) * 255).astype(np.uint8))
    
    # Save depth
    if depth_np is not None:
        _save_depth16(ep_dir / f"ep{ep:03d}_depth.png", depth_np)
    
    # Save normal map with validation
    if normal_map_np is not None:
        if normal_map_np.shape[-1] == 3:  # Final safety check
            _save_png(ep_dir / f"ep{ep:03d}_normal.png",
                     (np.clip(normal_map_np, 0, 1) * 255).astype(np.uint8))
        else:
            print(f"[WARN] Skipping normal map save due to invalid shape: {normal_map_np.shape}")


def save_episode_comparisons(
    ep: int,
    ep_dir: Path,
    x: Any,
    mu_np: np.ndarray,
    tgt: np.ndarray,
    rs_full: Dict
):
    """Save comparison visualizations."""
    if torch.is_tensor(x):
        x_np = x.cpu().numpy()
    else:
        x_np = x
    
    save_comparison_png(
        ep_dir / f"ep{ep:03d}_comparison.png", 
        current_before=x_np, 
        current_after=mu_np,
        radial_after=mu_np, 
        target_before=tgt,
        dpi=rs_full.get("png", {}).get("dpi", 160),
        ptsize=rs_full.get("png", {}).get("ptsize", 0.5)
    )
    
    save_axis_hist_png(
        ep_dir / f"ep{ep:03d}_axis_hist.png", 
        mu_np,
        dpi=rs_full.get("png", {}).get("dpi", 160)
    )


def save_episode_summary(
    ep: int,
    ep_dir: Path,
    F: Any,
    mu_np: np.ndarray,
    loss_physics: float
):
    """Save episode summary JSON."""
    if torch.is_tensor(F):
        F_np = F.cpu().numpy()
    else:
        F_np = F
    
    J = np.linalg.det(F_np)
    
    summary = {
        "episode": ep + 1,
        "J_min": float(J.min()),
        "J_mean": float(J.mean()),
        "loss_physics_final": float(loss_physics),
        "num_surface_points": len(mu_np),
    }
    
    with (ep_dir / f"ep{ep:03d}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_episode_data(
    ep: int,
    ep_dir: Path,
    mu_np: np.ndarray,
    cov_np: np.ndarray,
    particle_color: list
):
    """Save episode data files (NPZ, PLY)."""
    rgb_mu = np.tile(np.array(particle_color, dtype=np.float32), (len(mu_np), 1))
    
    save_gaussians_npz(
        ep_dir / f"ep{ep:03d}_gaussians.npz", 
        mu_np, cov_np, rgb=rgb_mu
    )
    
    save_ply_xyz(
        ep_dir / f"ep{ep:03d}_surface_{len(mu_np)}.ply", 
        mu_np
    )


# ============================================================================
# E2E Training Loop
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
    tgt: np.ndarray
) -> Dict:
        """Run a single episode with E2E training."""
        print(f"\n{'='*70}")
        print(f"Episode {ep+1} START")
        print(f"{'='*70}")
        
        # Setup
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
        
        for pass_idx in range(num_passes):
            print(f"\n{'─'*70}")
            print(f"Pass {pass_idx+1}/{num_passes}")
            print(f"{'─'*70}")
            
            # Phase 1: Inject previous render gradients
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
            
            # Phase 2: Physics optimization
            loss_physics = run_physics_optimization(
                cg, opt, num_timesteps, control_stride, ep,
                pass_idx=pass_idx  
            )
            
            # Phase 3: Compute new render gradients
            seed = 9999 + ep*1000 + pass_idx
            
            print(f"[Render] Computing loss for Pass {pass_idx+1}...")
            result = compute_render_loss_pass(
                cg, num_timesteps, rs_full, ema_state, renderer,
                loss_manager, target_render, view_params, campos,
                render_cfg, particle_color, seed
            )
            
            if result[0] is not None:
                ema_state, F, x = result
                
                # Extract gradients
                accumulated_render_grads = extract_render_gradients(F, x)
                
                if accumulated_render_grads:
                    print(f"├─ ✅ Render grads saved for Pass {pass_idx+2}")
                    print(f"├─ dLdF shape: {accumulated_render_grads['dLdF'].shape}")
                    print(f"└─ dLdx shape: {accumulated_render_grads['dLdx'].shape}\n")
                else:
                    print(f"└─ ⚠️ Gradient extraction failed (F.grad is None)\n")
            else:
                print(f"└─ ⚠️ compute_render_loss_pass returned None\n")
            
            # Phase 4: Visualization (last pass only)
            if pass_idx == num_passes - 1:
                print(f"[Visualization] Saving final results...")
                seed = 9999 + ep*1000 + pass_idx
                visualize_episode(
                    ep, out_dir, cg, num_timesteps, rs_full, ema_state,
                    renderer, campos, render_cfg, particle_color,
                    png_enabled, tgt, loss_physics, seed
                )
        
        # Cleanup
        print(f"\n{'='*70}")
        print(f"Episode {ep+1} COMPLETE")
        print(f"{'='*70}\n")
        
        accumulated_render_grads = None
        if cg.has_render_gradients():
            cg.clear_render_gradients()
        
        return ema_state


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main entry point."""
    import yaml
    
    # Parse arguments
    ap = argparse.ArgumentParser(description="DiffMPM E2E + Silhouette + Curvature")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--png", action="store_true", help="Export PNGs.")
    ap.add_argument("--png-dpi", type=int, default=160)
    ap.add_argument("--png-ptsize", type=float, default=0.6)
    ap.add_argument("--e2e", action="store_true", help="Enable E2E training mode.")
    args = ap.parse_args()
    
    # Load configuration
    print("[Config] Loading configuration...")
    cfg, cfg_dir = load_config(args.config)
    validate_config(cfg)
    
    out_dir = Path(cfg.get("output_dir", "output/runtime_surface"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build optimization input
    print("[Config] Building optimization parameters...")
    opt = build_opt_input(cfg)
    
    # Initialize point clouds and grids
    print("[Init] Loading point clouds...")
    input_pc, target_pc = initialize_point_clouds(opt)
    
    print("[Init] Initializing grids...")
    input_grid, target_grid = initialize_grids(opt)
    
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    
    # Create computation graph
    print("[Init] Creating computation graph...")
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)
    tgt = _np(diffmpm_bindings.get_positions_from_pc(target_pc))
    
    # Runtime surface config
    print("[Config] Setting up runtime surface...")
    rs = default_cfg()
    rs_user = cfg.get("sampling", {}).get("runtime_surface", {}) or {}
    rs.update(rs_user)
    rs.setdefault("png", {"enabled": True, "dpi": 160, "ptsize": 0.5})
    
    if args.png:
        rs["png"]["enabled"] = True
        rs["png"]["dpi"] = args.png_dpi
        rs["png"]["ptsize"] = args.png_ptsize
    
    # Render config
    render_cfg = cfg.get("render", {}) or {}
    particle_color = render_cfg.get("particle_color", DEFAULT_PARTICLE_COLOR)
    
    # Camera and renderer
    print("[Init] Setting up renderer...")
    cam_cfg = cfg.get("camera", {}) or {}
    renderer, view_params = setup_renderer(cam_cfg, render_cfg)
    HAVE_3DGS = renderer is not None
    
    # E2E setup
    enable_e2e = args.e2e or cfg.get("optimization", {}).get("loss", {}).get("enabled", False)
    loss_manager = None
    target_render = None
    
    if enable_e2e and E2ELossManager is not None and HAVE_3DGS:
        print("\n" + "="*70)
        print("🚀 E2E INTERLEAVED + SILHOUETTE + CURVATURE MODE")
        print("="*70)
        
        loss_config = cfg.get("optimization", {}).get("loss", {})
        loss_manager = E2ELossManager(loss_config)
        print(f"  Loss weights: {loss_manager.get_weights()}")
        
        print("\n[E2E] Creating target render (with Σ★)...")
        target_render = create_target_render(
            target_pc, renderer, rs, view_params['campos'],
            render_cfg, particle_color, out_dir / "target", view_params['view_T']
        )
        print("="*70 + "\n")
    else:
        if enable_e2e:
            print("[WARN] E2E mode requested but dependencies missing.")
        enable_e2e = False
    
    png_enabled = rs.get("png", {}).get("enabled", True) or args.png
    
    # Main optimization loop
    ema_state = {}
    
    for ep in range(int(opt.num_animations)):
        if enable_e2e and loss_manager is not None and target_render is not None:
            num_passes = 3
            num_timesteps = int(opt.num_timesteps)       
            control_stride = int(opt.control_stride) 
            
            ema_state = run_e2e_episode(
                ep, cg, opt, num_timesteps, control_stride, num_passes,
                rs, ema_state, renderer, loss_manager, target_render,
                view_params, view_params['campos'], render_cfg, particle_color,
                out_dir, png_enabled, tgt
            )
        else:
            # Standard physics-only optimization
            print(f"\n[Episode {ep+1}] Running standard physics optimization...")
            opt.current_episodes = ep
            cg.run_optimization(opt)
        
        # Promote to next episode
        if ep < int(opt.num_animations) - 1:
            print(f"\n[Promote] Moving final state to next episode...")
            cg.promote_last_as_initial()
    
    print("\n" + "="*70)
    print("All episodes finished.")
    print("="*70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())