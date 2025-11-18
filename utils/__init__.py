"""
Utilities Package

Modularized utilities for PhysMorph-GS pipeline.
"""

from .physics_utils import (
    initialize_point_clouds,
    initialize_grids,
    initialize_comp_graph,
    build_opt_input,
    run_physics_optimization,
    extract_target_point_cloud,
)

from .rendering_utils import (
    setup_renderer,
    upsample_target,
    render_target,
    normalize_render_outputs,
    create_target_render,
    prepare_rendering_inputs,
    upsample_current_state,
    compute_render_loss_pass,
    extract_render_gradients,
)

from .io_utils import (
    save_image_png,
    save_depth_png,
    save_target_renders,
    save_gaussians_npz,
    save_point_cloud_ply,
    save_episode_summary,
    save_episode_data,
)

from visualization.utils import (
    save_episode_images,
    save_episode_comparisons,
    create_axis_histogram,
    visualize_episode,
)

__all__ = [
    # Physics
    'initialize_point_clouds',
    'initialize_grids',
    'initialize_comp_graph',
    'build_opt_input',
    'run_physics_optimization',
    'extract_target_point_cloud',
    
    # Rendering
    'setup_renderer',
    'upsample_target',
    'render_target',
    'normalize_render_outputs',
    'create_target_render',
    'prepare_rendering_inputs',
    'upsample_current_state',
    'compute_render_loss_pass',
    'extract_render_gradients',
    
    # I/O
    'save_image_png',
    'save_depth_png',
    'save_target_renders',
    'save_gaussians_npz',
    'save_point_cloud_ply',
    'save_episode_summary',
    'save_episode_data',
    
    # Visualization
    'save_episode_images',
    'save_episode_comparisons',
    'create_axis_histogram',
    'visualize_episode',
]
