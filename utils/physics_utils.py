"""
Physics Utilities - MPM Simulation & Optimization

MPM (Material Point Method) simulation wrapper and optimization utilities.
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional


# ============================================================================
# Physics Initialization
# ============================================================================

def initialize_point_clouds(opt: Any, cfg: Optional[Dict] = None) -> Tuple[Any, Any]:
    """
    Initialize input and target point clouds from meshes.
    
    Args:
        opt: Optimization configuration with paths
    
    Returns:
        Tuple of (input_pc, target_pc)
    """
    import diffmpm_bindings
    
    sim_cfg = (cfg or {}).get("simulation", {})
    shell_cfg = sim_cfg.get("shell_sampling", {}) or {}
    shell_enabled = bool(shell_cfg.get("enabled", False))
    apply_jitter = bool(shell_cfg.get("apply_jitter", True))

    if shell_enabled:
        surface_ppc = int(shell_cfg.get("surface_points_per_cell_cuberoot", shell_cfg.get("surface_ppc", max(int(opt.points_per_cell_cuberoot), 6))))
        interior_ppc = int(shell_cfg.get("interior_points_per_cell_cuberoot", shell_cfg.get("interior_ppc", max(2, int(opt.points_per_cell_cuberoot) - 1))))
        shell_thickness_cells = float(shell_cfg.get("shell_thickness_cells", 1.5))
        print(
            f"[Init] Shell-biased sampling enabled: "
            f"surface_ppc={surface_ppc}, interior_ppc={interior_ppc}, "
            f"shell_thickness_cells={shell_thickness_cells}"
        )
        input_pc = diffmpm_bindings.load_shell_biased_point_cloud_from_obj(
            opt.mpm_input_mesh_path, opt, surface_ppc, interior_ppc, shell_thickness_cells, apply_jitter
        )
        target_pc = diffmpm_bindings.load_shell_biased_point_cloud_from_obj(
            opt.mpm_target_mesh_path, opt, surface_ppc, interior_ppc, shell_thickness_cells, apply_jitter
        )
    else:
        # Apply jitter to both input and target for consistent particle distribution
        input_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_input_mesh_path, opt, apply_jitter=True)
        target_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_target_mesh_path, opt, apply_jitter=True)
    
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
        int((opt.grid_max_point[i] - opt.grid_min_point[i]) / dx)
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

    # Adaptive alpha parameters (with backward-compatible defaults)
    opt.adaptive_alpha_enabled = bool(opt_cfg.get("adaptive_alpha_enabled", True))
    opt.adaptive_alpha_target_norm = float(opt_cfg.get("adaptive_alpha_target_norm", 2500.0))
    opt.adaptive_alpha_min_scale = float(opt_cfg.get("adaptive_alpha_min_scale", 0.1))
    
    return opt


__all__ = [
    'initialize_point_clouds',
    'initialize_grids',
    'initialize_comp_graph',
    'build_opt_input',
]
