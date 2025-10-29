"""
Physics Utilities - MPM Simulation & Optimization

MPM (Material Point Method) simulation wrapper and optimization utilities.
"""

import numpy as np
from typing import Any, Tuple, Dict


# ============================================================================
# Physics Initialization
# ============================================================================

def initialize_point_clouds(opt: Any) -> Tuple[Any, Any]:
    """
    Initialize input and target point clouds from meshes.
    
    Args:
        opt: Optimization configuration with paths
    
    Returns:
        Tuple of (input_pc, target_pc)
    """
    import diffmpm_bindings
    
    input_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_input_mesh_path, opt)
    target_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_target_mesh_path, opt)
    
    return input_pc, target_pc


def initialize_grids(opt: Any) -> Tuple[Any, Any]:
    """
    Initialize MPM grids for simulation.
    
    Args:
        opt: Optimization configuration with grid parameters
    
    Returns:
        Tuple of (input_grid, target_grid)
    """
    import diffmpm_bindings
    
    print("[Init] Initializing grids...")
    
    # Calculate grid dimensions
    dx = opt.grid_dx
    grid_size = [
        int((opt.grid_max_point[i] - opt.grid_min_point[i]) / dx) + 1
        for i in range(3)
    ]
    
    # Create Grid objects
    input_grid = diffmpm_bindings.Grid(
        grid_size[0], grid_size[1], grid_size[2],
        dx,
        opt.grid_min_point
    )
    
    target_grid = diffmpm_bindings.Grid(
        grid_size[0], grid_size[1], grid_size[2],
        dx,
        opt.grid_min_point
    )
    
    print(f"Generated grid of size: {grid_size[0]}x{grid_size[1]}x{grid_size[2]} ({grid_size[0]*grid_size[1]*grid_size[2]} nodes)")
    
    return input_grid, target_grid


def initialize_comp_graph(
    input_pc: Any,
    input_grid: Any,
    target_grid: Any
) -> Any:
    """
    Initialize computation graph for MPM simulation.
    
    Args:
        input_pc: Input point cloud
        input_grid: Input grid
        target_grid: Target grid (for loss computation)
    
    Returns:
        CompGraph instance
    """
    import diffmpm_bindings
    
    # CompGraph(PointCloud, Grid, const Grid) - 3 arguments
    cg = diffmpm_bindings.CompGraph(input_pc, input_grid, target_grid)
    
    return cg


def build_opt_input(cfg: Dict) -> Any:
    """
    Build optimization input from configuration.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        OptInput object with all simulation parameters
    """
    import diffmpm_bindings
    
    sim_cfg = cfg.get("simulation", {})
    opt_cfg = cfg.get("optimization", {})
    
    opt = diffmpm_bindings.OptInput()
    
    # I/O paths (using names defined in bind.cpp)
    opt.mpm_input_mesh_path = cfg.get("input_mesh_path", "")
    opt.mpm_target_mesh_path = cfg.get("target_mesh_path", "")
    
    # Grid configuration
    opt.grid_dx = float(sim_cfg.get("grid_dx", 0.75))
    opt.points_per_cell_cuberoot = int(sim_cfg.get("points_per_cell_cuberoot", 3))
    
    grid_min = sim_cfg.get("grid_min_point", [-16.0, -16.0, -16.0])
    grid_max = sim_cfg.get("grid_max_point", [16.0, 16.0, 16.0])
    opt.grid_min_point = tuple(float(x) for x in grid_min)
    opt.grid_max_point = tuple(float(x) for x in grid_max)
    
    # Material properties
    opt.lam = float(sim_cfg.get("lam", 38888.89))
    opt.mu = float(sim_cfg.get("mu", 58333.3))
    opt.p_density = float(sim_cfg.get("density", 75.0))
    
    # Simulation parameters
    opt.dt = float(sim_cfg.get("dt", 0.00833333333))
    opt.drag = float(sim_cfg.get("drag", 0.5))
    opt.smoothing_factor = float(sim_cfg.get("smoothing_factor", 0.955))
    
    external_force = sim_cfg.get("external_force", [0.0, 0.0, 0.0])
    opt.f_ext = tuple(float(x) for x in external_force)
    
    # Optimization parameters
    opt.num_animations = int(opt_cfg.get("num_animations", 50))
    opt.num_timesteps = int(opt_cfg.get("num_timesteps", 10))
    opt.control_stride = int(opt_cfg.get("control_stride", 1))
    opt.max_gd_iters = int(opt_cfg.get("max_gd_iters", 1))
    opt.max_ls_iters = int(opt_cfg.get("max_ls_iters", 10))
    opt.initial_alpha = float(opt_cfg.get("initial_alpha", 0.01))
    opt.gd_tol = float(opt_cfg.get("gd_tol", 0.0001))
    opt.current_episodes = 0
    
    return opt


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
    """
    Run physics optimization over control timesteps using Adam.
    
    For each control timestep t:
      1. Forward: Simulate x(t) → x(t+1) using MPM
      2. Loss: L_physics = ||x(T) - x_target||² (mass matching)
      3. Backward: Compute ∂L_total/∂control(t)
      4. Update: Adam step on control forces
    
    Args:
        cg: Computation graph (C++ MPM backend)
        opt: Optimization configuration
        num_timesteps: Total number of timesteps
        control_stride: Control frame interval (usually 1)
        ep: Current episode number
        pass_idx: Current pass number (for logging)
    
    Returns:
        loss_physics: Final physics loss value
    """
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


def extract_target_point_cloud(target_pc: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract positions and deformation gradients from target point cloud.
    
    Args:
        target_pc: Target point cloud from MPM
    
    Returns:
        Tuple of (x_tgt, F_tgt)
    """
    def _np(x):
        """Convert to numpy."""
        if hasattr(x, 'cpu'):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)
    
    try:
        x_tgt = target_pc.get_positions_torch(requires_grad=False)
        F_tgt = target_pc.get_def_grads_total_torch(requires_grad=False)
    except AttributeError:
        x_tgt = _np(target_pc.get_positions())
        F_tgt = _np(target_pc.get_def_grads_total())
    
    return x_tgt, F_tgt


__all__ = [
    'initialize_point_clouds',
    'initialize_grids',
    'initialize_comp_graph',
    'build_opt_input',
    'run_physics_optimization',
    'extract_target_point_cloud',
]

